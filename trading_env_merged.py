import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from server import ZMQRepServer
except Exception:
    ZMQRepServer = None


class TradingEnv(gym.Env):
    """
    Merged DQN-ready trading environment.

    Key features:
      - Gymnasium API:
            reset(*, seed, options) -> (obs, info)
            step(action) -> (obs, reward, terminated, truncated, info)
      - Discrete actions: 0=HOLD, 1=BUY, 2=SELL
      - Per-trade: BUY/SELL opens & closes within same step (always flat after step)
      - Tick-driven TP/SL exits, EOF treated as SL
      - Train / eval split with random starting ticks
      - Optional normalization:
            * bar features (via precomputed or on-the-fly scaler)
            * ask / bid / spread
      - Reward modes:
            "risk" : profit normalized by risk_per_trade_usd (or 1R) minus DD increment penalty
            "pnl"  : profit in USD normalized by risk_per_trade_usd minus DD increment penalty
      - Detailed info dict for logging & debugging
      - Optional ZeroMQ server integration
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pip_decimal: float,
        candles_file: str,
        tick_file: str,
        bind_address: str = "tcp://*:5555",
        cache_dir: str = "cache_fx_EURUSD_D1",

        # Trading config
        tp_pips: float = 50.0,
        sl_pips: float = 50.0,
        lot: float = 1.0,
        start_balance: float = 100_000.0,
        reset_balance_each_episode: bool = True,
        include_spread_cost: bool = False,
        exchange_rate: float = 1.0,   # 1.0 for USD-quoted symbols like EURUSD

        # Risk & penalties
        dd_penalty_lambda: float = 1.0,
        max_dd_stop: float = 0.30,

        # Reward configuration
        reward_mode: str = "risk",    # "risk" or "pnl"
        risk_per_trade_usd: float = 1000.0,

        # Episode / data split
        max_steps_per_episode: int | None = 5000,
        train_fraction: float = 0.7,
        eval_mode: bool = False,

        # Observation
        window_len: int = 32,

        # Normalization
        normalize_prices: bool = True,
        normalize_bars: bool = True,

        # Runtime / integration
        start_server: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        # Paths (kept for compatibility / future use)
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.cache_dir = cache_dir

        # Trading config
        self.pip_decimal = float(pip_decimal)
        self.tp_pips = float(tp_pips)
        self.sl_pips = float(sl_pips)
        self.lot = float(lot)
        self.start_balance = float(start_balance)
        self.reset_balance_each_episode = bool(reset_balance_each_episode)
        self.include_spread_cost = bool(include_spread_cost)
        self.exchange_rate = float(exchange_rate)

        # Risk & penalties
        self.dd_penalty_lambda = float(dd_penalty_lambda)
        self.max_dd_stop = float(max_dd_stop)

        # Reward
        self.reward_mode = str(reward_mode)
        self.risk_per_trade_usd = float(risk_per_trade_usd)

        # Episode / split
        self.max_steps_per_episode = int(max_steps_per_episode) if max_steps_per_episode is not None else None
        self.train_fraction = float(train_fraction)
        self.eval_mode = bool(eval_mode)

        # Observation config
        self.window_len = int(window_len)

        # Normalization flags
        self.normalize_prices = bool(normalize_prices)
        self.normalize_bars = bool(normalize_bars)

        # RNG
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        # Load cache (depends on train_fraction)
        self._load_cache(self.cache_dir)

        # Gym spaces: obs = window_len * n_feats + [ask, bid, spread]
        self.obs_dim = self.window_len * self.n_feats + 3
        high = np.full((self.obs_dim,), 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=HOLD,1=BUY,2=SELL

        # Account / episode state (initialized in reset)
        self.balance = self.start_balance
        self.equity = self.start_balance
        self.equity_peak = self.start_balance
        self.max_drawdown = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0
        self.t = 0
        self.current_tick_row = 0

        # Normalization anchors (set in reset)
        self.ask_mean = self.ask_std = None
        self.bid_mean = self.bid_std = None
        self.spread_mean = self.spread_std = None

        # Optional ZMQ server
        self.server = None
        if start_server and ZMQRepServer is not None:
            self.server = ZMQRepServer(bind_address, self._handle_request)

    # -------------------------------------------------------------------------
    # ZeroMQ hooks
    # -------------------------------------------------------------------------
    def start_server(self):
        if self.server is not None:
            print("Starting Env Server...")
            self.server.start()

    def stop_server(self):
        if self.server is not None:
            print("Stopping Env Server...")
            self.server.stop()

    def _handle_request(self, request):
        """
        Minimal ZMQ handler. Expected payload:
            {"cmd": "HOLD"|"BUY"|"SELL"}
        Returns step info in a compact dict.
        """
        if not isinstance(request, dict):
            return {"reply": "invalid_request"}

        cmd = request.get("cmd")
        mapping = {"HOLD": 0, "BUY": 1, "SELL": 2}
        if cmd in mapping:
            obs, reward, terminated, truncated, info = self.step(mapping[cmd])
            return {
                "obs": obs.tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        return {"reply": "unknown_command", "received": request}

    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------
    def seed(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        # Balance handling
        if self.reset_balance_each_episode or not hasattr(self, "balance"):
            self.balance = self.start_balance
        # else: keep running balance across episodes

        # Reset state
        self.equity = float(self.balance)
        self.equity_peak = float(self.balance)
        self.max_drawdown = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0

        # Choose starting tick with warmup + train/eval split
        min_bar = self.window_len - 1
        base = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        if base >= self.n_ticks:
            raise ValueError("Warmup exceeds available ticks; check window_len or data.")

        split_tick = int(self.n_ticks * self.train_fraction)

        if self.eval_mode:
            # Start in unseen (evaluation) region
            start_low = max(base, split_tick)
            start_high = max(start_low + 1, self.n_ticks - 1)
        else:
            # Start in training region
            start_low = base
            start_high = max(start_low + 1, max(base + 1, split_tick))

        if start_low >= start_high:
            # Fallback: just use base
            self.t = base
        else:
            self.t = int(self._rng.integers(low=start_low, high=start_high, endpoint=False))

        self.current_tick_row = self.t

        # Fit price normalizers from a rolling window behind t
        lookback = max(0, self.t - 20_000)
        if self.normalize_prices and self.t > lookback:
            ask_slice = np.asarray(self.tick_ask[lookback:self.t + 1], dtype=np.float32)
            bid_slice = np.asarray(self.tick_bid[lookback:self.t + 1], dtype=np.float32)
            spread_slice = ask_slice - bid_slice

            self.ask_mean = float(ask_slice.mean())
            self.ask_std = float(ask_slice.std() + 1e-8)
            self.bid_mean = float(bid_slice.mean())
            self.bid_std = float(bid_slice.std() + 1e-8)
            self.spread_mean = float(spread_slice.mean())
            self.spread_std = float(spread_slice.std() + 1e-8)
        else:
            self.ask_mean = self.ask_std = None
            self.bid_mean = self.bid_std = None
            self.spread_mean = self.spread_std = None

        obs = self._get_observation(self.t)
        return obs, {}

    def step(self, action: int):
        # If episode is already over, no-op
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, self.terminated, self.truncated, {
                "note": "episode already ended"
            }

        # End-of-data guard
        if self.t >= self.n_ticks - 1:
            self.terminated = True
            info = {"reason": "eof"}
            return self._get_observation(self.t), 0.0, True, self.truncated, info

        # Map action id -> label
        if action == 0:
            act = "HOLD"
        elif action == 1:
            act = "BUY"
        elif action == 2:
            act = "SELL"
        else:
            act = "HOLD"

        # HOLD: move forward one tick, zero reward
        if act == "HOLD":
            self.t += 1
            self.current_tick_row = self.t
            self.steps += 1

            if self.max_steps_per_episode is not None and self.steps >= self.max_steps_per_episode:
                self.truncated = True

            obs = self._get_observation(self.t)
            info = {
                "action": "HOLD",
                "balance": float(self.balance),
                "equity": self._equity_mtm(),
                "equity_peak": float(self.equity_peak),
                "max_drawdown": float(self.max_drawdown),
                "closed_trade": False,
            }
            return obs, 0.0, False, self.truncated, info

        # ------------------------------------------------------------------
        # BUY / SELL: per-trade simulation to TP or SL (or EOF -> SL)
        # ------------------------------------------------------------------
        entry_idx = self.t
        entry_ask = float(self.tick_ask[entry_idx])
        entry_bid = float(self.tick_bid[entry_idx])
        spread_once_pips = max(0.0, (entry_ask - entry_bid) / self.pip_decimal)

        tp_value = sl_value = None
        is_profitable = None
        k = entry_idx

        if act == "BUY":
            tp_value = entry_ask + self.tp_pips * self.pip_decimal
            sl_value = entry_bid - self.sl_pips * self.pip_decimal
            # simulate on BID for long exit
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_bid[k])
                if px >= tp_value:
                    is_profitable = True
                    break
                if px <= sl_value:
                    is_profitable = False
                    break
            close_px = float(self.tick_bid[k])
        else:  # SELL
            tp_value = entry_bid - self.tp_pips * self.pip_decimal
            sl_value = entry_ask + self.sl_pips * self.pip_decimal
            # simulate on ASK for short exit
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_ask[k])
                if px <= tp_value:
                    is_profitable = True
                    break
                if px >= sl_value:
                    is_profitable = False
                    break
            close_px = float(self.tick_ask[k])

        # Move time cursor to close tick
        self.t = k
        self.current_tick_row = self.t

        # EOF without TP/SL: treat as SL (loss)
        if is_profitable is None:
            is_profitable = False
            if self.t >= self.n_ticks - 1:
                self.terminated = True

        # Fixed ±TP/SL pips for PnL, then apply spread once if desired
        base_pips = self.tp_pips if is_profitable else -self.sl_pips
        pnl_pips = float(base_pips)
        if self.include_spread_cost and spread_once_pips > 0.0:
            pnl_pips -= spread_once_pips

        pip_value = self._pip_value_usd()
        profit_usd = float(pnl_pips * pip_value)

        trade_duration_ticks = int(self.t - entry_idx)

        # ----------------- Update account & drawdown -----------------
        next_balance = self.balance + profit_usd

        if next_balance <= 0.0:
            self.balance = 0.0
            self.equity = 0.0
            self.terminated = True
            self.truncated = True
        else:
            self.balance = float(next_balance)
            self.equity = self._equity_mtm()

        # Peak & drawdown
        if self.equity > self.equity_peak:
            self.equity_peak = float(self.equity)
        if self.equity_peak > 0.0:
            dd_now = max(0.0, (self.equity_peak - self.equity) / self.equity_peak)
        else:
            dd_now = 0.0

        new_dd_increment = max(0.0, dd_now - self.max_drawdown)
        if dd_now > self.max_drawdown:
            self.max_drawdown = float(dd_now)

        if self.max_drawdown >= self.max_dd_stop:
            self.terminated = True
            self.truncated = True

        # ----------------- Reward computation -----------------
        reward = self._calculate_reward(
            profit_usd=profit_usd,
            new_dd_increment=new_dd_increment,
        )

        self.steps += 1
        if self.max_steps_per_episode is not None and self.steps >= self.max_steps_per_episode:
            self.truncated = True

        terminated_flag = bool(self.terminated)

        # 1R reference for logging
        one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
        profit_R = float(profit_usd / one_R_usd)

        info = {
            "action": act,
            "tp": float(tp_value),
            "sl": float(sl_value),
            "final_idx": int(self.t),
            "balance": float(self.balance),
            "equity": float(self.equity),
            "equity_peak": float(self.equity_peak),
            "dd_now": float(dd_now),
            "max_drawdown": float(self.max_drawdown),
            "new_dd_increment": float(new_dd_increment),
            "is_profitable": bool(is_profitable),
            "last_reward": float(reward),
            "profit": float(profit_usd),
            "profit_R": float(profit_R),
            "pnl_pips": float(pnl_pips),
            "close_price": float(close_px),
            "trade_duration_ticks": trade_duration_ticks,
            "eof": bool(self.t >= self.n_ticks - 1),
            "closed_trade": True,
            "reward_mode": self.reward_mode,
        }

        return self._get_observation(self.t), float(reward), terminated_flag, bool(self.truncated), info

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _pip_value_usd(self) -> float:
        """
        Value of 1 pip (pip_decimal) in USD for the configured lot size.
        Adjusts by exchange_rate when base/quote differ from USD.
        """
        return ((self.lot * 100000.0) * self.pip_decimal) / max(1e-12, self.exchange_rate)

    def _equity_mtm(self) -> float:
        """
        Mark-to-market equity. In this per-trade env we are flat after each step,
        so equity == balance, but kept as separate hook.
        """
        return float(self.balance)

    def _price_norm(self, x: float, mean: float, std: float) -> float:
        if not self.normalize_prices or mean is None or std is None or std == 0.0:
            return float(x)
        return float((x - mean) / (std + 1e-8))

    def _get_observation(self, tick_idx: int):
        """
        Observation vector:
            [window_len * normalized_bar_features] + [ask, bid, spread]
        """
        bar_idx = int(self.tick_to_bar[tick_idx])
        L = self.window_len
        start = bar_idx - L + 1

        if start < 0:
            pad = np.zeros((-start, self.n_feats), dtype=np.float32)
            win = np.asarray(self.bar_features[0:bar_idx + 1], dtype=np.float32)
            feat_win = np.vstack([pad, win])
        else:
            feat_win = np.asarray(self.bar_features[start:bar_idx + 1], dtype=np.float32)

        # Normalize bar features if configured and scaler available
        if self.normalize_bars and hasattr(self, "bars_mean") and self.bars_mean is not None:
            feat_win = (feat_win - self.bars_mean) / (self.bars_std + 1e-8)

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        a = self._price_norm(ask, self.ask_mean, self.ask_std)
        b = self._price_norm(bid, self.bid_mean, self.bid_std)
        s = self._price_norm(spread, self.spread_mean, self.spread_std)

        obs = np.hstack([feat_win.ravel(), np.array([a, b, s], dtype=np.float32)]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_reward(self, profit_usd: float, new_dd_increment: float) -> float:
        """
        Unified reward:
          - If reward_mode == "risk":
                reward = (profit_usd / denom) - λ * ΔDD
                denom = risk_per_trade_usd (if >0) else 1R (pip_value * sl_pips)
          - If reward_mode == "pnl":
                reward = (profit_usd / denom) - λ * ΔDD
                denom = risk_per_trade_usd (if >0) else 1.0
          - Else:
                fallback to 1R-based normalization.
        """
        pip_value = self._pip_value_usd()
        one_R_usd = max(1e-8, pip_value * self.sl_pips)

        mode = (self.reward_mode or "risk").lower()

        if mode == "risk":
            denom = self.risk_per_trade_usd if self.risk_per_trade_usd > 0 else one_R_usd
        elif mode == "pnl":
            denom = self.risk_per_trade_usd if self.risk_per_trade_usd > 0 else 1.0
        else:
            denom = one_R_usd

        denom = max(1e-8, denom)
        pnl_norm = float(profit_usd) / denom
        penalty = self.dd_penalty_lambda * float(new_dd_increment)
        return float(pnl_norm - penalty)

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    def _load_cache(self, cache_dir: str):
        needed = ["bars_features.npy", "bar_times.npy", "tick_ask.npy", "tick_bid.npy", "tick_to_bar.npy"]
        for f in needed:
            p = os.path.join(cache_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cache artifact: {p}")

        self.bar_features = np.load(os.path.join(cache_dir, "bars_features.npy"), mmap_mode="r")
        self.bar_times = np.load(os.path.join(cache_dir, "bar_times.npy"), mmap_mode="r")
        self.tick_ask = np.load(os.path.join(cache_dir, "tick_ask.npy"), mmap_mode="r")
        self.tick_bid = np.load(os.path.join(cache_dir, "tick_bid.npy"), mmap_mode="r")
        self.tick_to_bar = np.load(os.path.join(cache_dir, "tick_to_bar.npy"), mmap_mode="r")

        self.n_bars = int(self.bar_features.shape[0])
        self.n_feats = int(self.bar_features.shape[1])
        self.n_ticks = int(self.tick_ask.shape[0])

        assert self.tick_bid.shape == self.tick_ask.shape
        assert self.tick_to_bar.shape[0] == self.n_ticks

        # Precompute bar feature scaler for train split if requested
        self.bars_mean = None
        self.bars_std = None
        if self.normalize_bars:
            mean_path = os.path.join(cache_dir, "bars_mean.npy")
            std_path = os.path.join(cache_dir, "bars_std.npy")

            if os.path.exists(mean_path) and os.path.exists(std_path):
                self.bars_mean = np.load(mean_path).astype(np.float32)
                self.bars_std = np.load(std_path).astype(np.float32)
            else:
                split_idx = int(self.n_bars * float(self.train_fraction))
                if split_idx <= 0 or split_idx > self.n_bars:
                    split_idx = self.n_bars
                train_feats = np.asarray(self.bar_features[:split_idx], dtype=np.float32)
                self.bars_mean = train_feats.mean(axis=0)
                self.bars_std = train_feats.std(axis=0) + 1e-8
