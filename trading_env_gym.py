
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    # optional: only needed if you actually run the ZMQ server
    from server import ZMQRepServer
except Exception:
    ZMQRepServer = None


class TradingEnv(gym.Env):
    """
    DQN-ready trading environment.

    Key changes vs original:
      - Gymnasium API: reset(seed, options) -> (obs, info); step -> (obs, reward, terminated, truncated, info)
      - Discrete actions: 0=HOLD, 1=BUY, 2=SELL
      - action_space / observation_space defined
      - Optional price normalization for [ask, bid, spread]
      - Episode cap via max_steps_per_episode
      - Train/eval split & random start tick
      - Reward modes: "risk" (normalized PnL - DD penalty) or "pnl" (true pips incl. spread)
      - Additional diagnostics in info: pnl_pips, drawdown metrics, etc.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        pip_decimal: float,
        candles_file: str,
        tick_file: str,
        bind_address: str = "tcp://*:5555",
        cache_dir: str = "cache_fx_EURUSD_D1",
        # New knobs
        max_steps_per_episode: int = 5000,
        reward_mode: str = "risk",          # "risk" (normalized pnl - dd penalty) or "pnl" (realized pips)
        train_fraction: float = 0.7,
        eval_mode: bool = False,
        normalize_prices: bool = True,
        dd_penalty_lambda: float = 1.0,
        risk_per_trade_usd: float = 1000.0,
        max_dd_stop: float = 0.30,
        window_len: int = 32,
        tp_pips: float = 50.0,
        sl_pips: float = 50.0,
        lot: float = 1.0,
        start_server: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        # --- I/O / data paths (kept for compatibility) ---
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.cache_dir = cache_dir

        # --- Trading & risk config ---
        self.pip_decimal = pip_decimal
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.lot = lot
        self.risk_per_trade_usd = risk_per_trade_usd
        self.dd_penalty_lambda = dd_penalty_lambda
        self.max_dd_stop = max_dd_stop

        # --- Episode control ---
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.reward_mode = reward_mode
        self.train_fraction = float(train_fraction)
        self.eval_mode = bool(eval_mode)
        self.normalize_prices = bool(normalize_prices)
        self.window_len = int(window_len)

        # --- Runtime state ---
        self.balance = 100000.0
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0
        self.truncated = False
        self.terminated = False
        self.position_open = False  # kept for completeness (we always flat in one step)
        self.t = 0
        self.current_tick_row = 0

        # --- Load cache data ---
        self._load_cache(self.cache_dir)

        # --- Gym spaces ---
        # obs: window_len * n_feats + [ask, bid, spread]
        self.obs_dim = self.window_len * self.n_feats + 3
        high = np.full((self.obs_dim,), 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL

        # --- RNG / seeding ---
        self.np_rng = np.random.default_rng(seed if seed is not None else 0)

        # --- Normalization anchors (set in reset) ---
        self.ask_mean = self.ask_std = None
        self.bid_mean = self.bid_std = None
        self.spread_mean = self.spread_std = None

        # --- Optional ZeroMQ server (unchanged behavior) ---
        self.server = None
        if start_server and ZMQRepServer is not None:
            self.server = ZMQRepServer(bind_address, self._handle_request)

    # ----------------------- External server hooks (optional) -----------------------
    def start_server(self):
        if self.server is not None:
            print("Starting Env Server...")
            self.server.start()

    def stop_server(self):
        if self.server is not None:
            print("Stopping Env Server...")
            self.server.stop()

    def _handle_request(self, request):
        """Optional ZMQ server handler; not used for in-process DQN training."""
        cmd = request.get("cmd") if isinstance(request, dict) else None
        mapping = {"HOLD": 0, "BUY": 1, "SELL": 2}
        if cmd in mapping:
            obs, reward, term, trunc, info = self.step(mapping[cmd])
            return {"obs": obs.tolist(), "reward": float(reward), "terminated": term, "truncated": trunc, "info": info}
        return {"reply": "unknown_command", "received": request}

    # ----------------------------- Gym API methods ---------------------------------
    def seed(self, seed: int | None = None):
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Account state
        self.balance = 100000.0
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0
        self.truncated = False
        self.terminated = False
        self.position_open = False
        self.steps = 0

        # Choose start tick (train/eval split) with warmup
        min_bar = self.window_len - 1
        base = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        split_tick = int(self.n_ticks * self.train_fraction)

        if self.eval_mode:
            start_low = max(base, split_tick)
            start_high = max(start_low + 1, self.n_ticks - 1)
            self.t = int(self.np_rng.integers(low=start_low, high=start_high, endpoint=False))
        else:
            start_low = base
            start_high = max(start_low + 1, split_tick - 1)
            self.t = int(self.np_rng.integers(low=start_low, high=start_high, endpoint=False))

        self.current_tick_row = self.t

        # Fit simple normalizers from a rolling window behind t
        lookback = max(0, self.t - 20000)
        if self.normalize_prices and self.t > lookback:
            ask_slice = np.asarray(self.tick_ask[lookback:self.t+1], dtype=np.float32)
            bid_slice = np.asarray(self.tick_bid[lookback:self.t+1], dtype=np.float32)
            spread_slice = ask_slice - bid_slice
            self.ask_mean, self.ask_std = float(ask_slice.mean()), float(ask_slice.std() + 1e-8)
            self.bid_mean, self.bid_std = float(bid_slice.mean()), float(bid_slice.std() + 1e-8)
            self.spread_mean, self.spread_std = float(spread_slice.mean()), float(spread_slice.std() + 1e-8)
        else:
            self.ask_mean = self.ask_std = None
            self.bid_mean = self.bid_std = None
            self.spread_mean = self.spread_std = None

        obs = self._get_observation(self.t)
        info = {}
        return obs, info

    def step(self, action: int):
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, self.terminated, self.truncated, {"note": "episode already ended"}

        # End of data guard
        if self.t >= self.n_ticks - 1:
            self.terminated = True
            return self._get_observation(self.t), 0.0, True, self.truncated, {"reason": "eof"}

        # Map action
        if action == 0:
            act = "HOLD"
        elif action == 1:
            act = "BUY"
        elif action == 2:
            act = "SELL"
        else:
            act = "HOLD"

        # HOLD: advance one tick
        if act == "HOLD":
            self.t += 1
            self.current_tick_row = self.t
            self.steps += 1
            if self.steps >= self.max_steps_per_episode:
                self.truncated = True
            return self._get_observation(self.t), 0.0, False, self.truncated, {"action": "HOLD"}

        # Entry prices
        entry_ask = float(self.tick_ask[self.t])
        entry_bid = float(self.tick_bid[self.t])
        spread_once = (entry_ask - entry_bid) / self.pip_decimal

        tp_value = sl_value = None
        is_profitable = None
        pnl_pips = 0.0

        k = self.t
        if act == "BUY":
            entry = entry_ask
            tp_value = entry + self.tp_pips * self.pip_decimal
            sl_value = entry - self.sl_pips * self.pip_decimal
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_bid[k])  # long closes at BID
                if px >= tp_value:
                    is_profitable = True; break
                if px <= sl_value:
                    is_profitable = False; break
            self.t = k
            pnl_pips = (float(self.tick_bid[self.t]) - entry) / self.pip_decimal - spread_once

        else:  # SELL
            entry = entry_bid
            tp_value = entry - self.tp_pips * self.pip_decimal
            sl_value = entry + self.sl_pips * self.pip_decimal
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_ask[k])  # short closes at ASK
                if px <= tp_value:
                    is_profitable = True; break
                if px >= sl_value:
                    is_profitable = False; break
            self.t = k
            pnl_pips = (entry - float(self.tick_ask[self.t])) / self.pip_decimal - spread_once

        # EOF without TP/SL
        if is_profitable is None:
            is_profitable = False
            if self.t >= self.n_ticks - 1:
                self.terminated = True

        # Reward + account updates
        reward = self._calc_reward(is_profitable, pnl_pips)

        self.steps += 1
        if self.steps >= self.max_steps_per_episode:
            self.truncated = True

        obs = self._get_observation(self.t)
        info = {
            "action": act,
            "tp": tp_value,
            "sl": sl_value,
            "final_idx": self.t,
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "max_drawdown": self.max_drawdown,
            "is_profitable": bool(is_profitable),
            "pnl_pips": float(pnl_pips),
            "eof": (self.t >= self.n_ticks - 1),
        }
        return obs, float(reward), self.terminated, self.truncated, info

    # ----------------------------- Helpers -----------------------------------------
    def _price_norm(self, x, mean, std):
        if not self.normalize_prices or mean is None or std is None or std == 0.0:
            return float(x)
        return float((x - mean) / (std + 1e-8))

    def _get_observation(self, tick_idx: int):
        bar_idx = int(self.tick_to_bar[tick_idx])
        L = self.window_len
        start = bar_idx - L + 1

        if start < 0:
            pad = np.zeros((-start, self.n_feats), dtype=np.float32)
            win = np.asarray(self.bar_features[0:bar_idx+1], dtype=np.float32)
            feat_win = np.vstack([pad, win])
        else:
            feat_win = np.asarray(self.bar_features[start:bar_idx+1], dtype=np.float32)

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        a = self._price_norm(ask, self.ask_mean, self.ask_std)
        b = self._price_norm(bid, self.bid_mean, self.bid_std)
        s = self._price_norm(spread, self.spread_mean, self.spread_std)

        obs = np.hstack([feat_win.ravel(), np.array([a, b, s], dtype=np.float32)])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _calc_reward(self, is_profitable: bool, pnl_pips: float):
        # pip value in USD (assumes quote currency USD; adjust exchange_rate if needed)
        pip_value = ((self.lot * 100000) * self.pip_decimal) / 1.0

        if self.reward_mode == "pnl":
            profit = pip_value * pnl_pips
        else:  # "risk" (default) uses fixed TP/SL outcome
            profit = pip_value * (self.tp_pips if is_profitable else -self.sl_pips)

        # Balance & equity
        self.balance = max(0.0, self.balance + profit)
        self.equity = self.balance
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        dd_now = 0.0 if self.equity_peak <= 0 else (self.equity_peak - self.equity) / (self.equity_peak + 1e-9)
        dd_now = max(0.0, dd_now)
        new_dd_increment = max(0.0, dd_now - self.max_drawdown)
        if dd_now > self.max_drawdown:
            self.max_drawdown = dd_now

        denom = max(1e-8, self.risk_per_trade_usd)
        pnl_norm = profit / denom
        penalty = self.dd_penalty_lambda * new_dd_increment
        reward = pnl_norm - penalty

        if self.max_drawdown >= self.max_dd_stop or self.balance <= 0.0:
            self.terminated = True
            self.truncated = True

        return float(reward)

    # ----------------------------- Data loading ------------------------------------
    def _load_cache(self, cache_dir: str):
        needed = ["bars_features.npy", "bar_times.npy", "tick_ask.npy", "tick_bid.npy", "tick_to_bar.npy"]
        for f in needed:
            p = os.path.join(cache_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cache artifact: {p}")

        self.bar_features = np.load(os.path.join(cache_dir, "bars_features.npy"), mmap_mode="r")
        self.bar_times    = np.load(os.path.join(cache_dir, "bar_times.npy"),    mmap_mode="r")
        self.tick_ask     = np.load(os.path.join(cache_dir, "tick_ask.npy"),     mmap_mode="r")
        self.tick_bid     = np.load(os.path.join(cache_dir, "tick_bid.npy"),     mmap_mode="r")
        self.tick_to_bar  = np.load(os.path.join(cache_dir, "tick_to_bar.npy"),  mmap_mode="r")

        self.n_bars  = int(self.bar_features.shape[0])
        self.n_feats = int(self.bar_features.shape[1])
        self.n_ticks = int(self.tick_ask.shape[0])

        assert self.tick_bid.shape == self.tick_ask.shape
        assert self.tick_to_bar.shape[0] == self.n_ticks
