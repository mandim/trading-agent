import os
import numpy as np
import pandas as pd  # (kept for possible CSV logging)
from server import ZMQRepServer

# ---- tiny “spaces” (no gym dependency) --------------------------------------
class _Discrete:
    def __init__(self, n):
        self.n = int(n)
    def sample(self):
        return int(np.random.randint(self.n))

class _Box:
    def __init__(self, shape, dtype=np.float32):
        self.shape = tuple(shape)
        self.dtype = dtype
# -----------------------------------------------------------------------------

class TradingEnv:
    """
    Tick-driven, per-trade environment with TP/SL exits.
    - Actions: "BUY", "SELL", "HOLD".
    - Trades close at first hit of TP or SL (or EOF -> treated as SL).
    - Reward = PnL normalized in R units (±1 when TP/SL are symmetric)
              − λ * (increment in drawdown).
    - Per-trade (instant-close) design: no open positions across steps.
    """

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
        reset_balance_each_episode: bool = True,  # training: True; long-book testing: False
        include_spread_cost: bool = False,
        exchange_rate: float = 1.0,  # use 1.0 for USD-quoted pairs like EURUSD

        # Risk & penalties
        dd_penalty_lambda: float = 1.0,  # 0.3–2.0 reasonable
        max_dd_stop: float = 0.30,       # hard stop at 30% drawdown

        # Observation / control
        window_len: int = 32,
        max_steps_per_episode: int | None = None,
        seed: int | None = None,
    ):
        # --- inputs / data handles
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.cache_dir = cache_dir

        # --- trading params
        self.pip_decimal = float(pip_decimal)
        self.tp_pips = float(tp_pips)
        self.sl_pips = float(sl_pips)
        self.lot = float(lot)
        self.start_balance = float(start_balance)
        self.reset_balance_each_episode = bool(reset_balance_each_episode)
        self.include_spread_cost = bool(include_spread_cost)
        self.exchange_rate = float(exchange_rate)

        # --- risk & penalties
        self.dd_penalty_lambda = float(dd_penalty_lambda)
        self.max_dd_stop = float(max_dd_stop)

        # --- obs/control setup
        self.window_len = int(window_len)
        self.max_steps_per_episode = max_steps_per_episode
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(1 << 31))

        # --- runtime state (set in reset)
        self.balance = self.start_balance
        self.equity = self.start_balance
        self.equity_peak = self.start_balance
        self.max_drawdown = 0.0
        self.truncated = False
        self.done = False
        self.t = 0  # tick cursor
        self.current_tick_row = 0
        self.episode_steps = 0

        # --- server
        self.server = ZMQRepServer(bind_address, self._handle_request)

        # --- load cached arrays (and define spaces)
        self._load_cache(self.cache_dir)
        obs_dim = self.window_len * self.n_feats + 3  # [bars window] + [ask,bid,spread]
        self.observation_space = _Box((obs_dim,), dtype=np.float32)
        self.action_space = _Discrete(3)  # 0=HOLD,1=BUY,2=SELL

    # --------------------- Server controls ---------------------
    def start_server(self):
        print("Starting Env Server...")
        self.server.start()

    def stop_server(self):
        print("Stopping Env Server...")
        self.server.stop()

    def _handle_request(self, request):
        """
        Minimal ZMQ handler (extend to return obs/reward/dones as needed in your client protocol).
        """
        cmd = request.get("cmd") if isinstance(request, dict) else None
        if cmd in ("BUY", "SELL", "HOLD"):
            obs, rew, done, trunc, info = self.step(cmd)
            return {"reply": f"{cmd} executed", "reward": rew, "done": done, "truncated": trunc, "info": info}
        else:
            return {"reply": "unknown_command", "received": request}

    # --------------------- Data loading ---------------------
    def _load_cache(self, cache_dir: str):
        needed = ["bars_features.npy", "bar_times.npy", "tick_ask.npy", "tick_bid.npy", "tick_to_bar.npy"]
        for f in needed:
            p = os.path.join(cache_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cache artifact: {p}")

        self.bar_features = np.load(os.path.join(cache_dir, "bars_features.npy"), mmap_mode="r")  # (n_bars, n_feats)
        self.bar_times    = np.load(os.path.join(cache_dir, "bar_times.npy"),    mmap_mode="r")   # (n_bars,)
        self.tick_ask     = np.load(os.path.join(cache_dir, "tick_ask.npy"),     mmap_mode="r")   # (n_ticks,)
        self.tick_bid     = np.load(os.path.join(cache_dir, "tick_bid.npy"),     mmap_mode="r")   # (n_ticks,)
        self.tick_to_bar  = np.load(os.path.join(cache_dir, "tick_to_bar.npy"),  mmap_mode="r")   # (n_ticks,)

        self.n_bars  = int(self.bar_features.shape[0])
        self.n_feats = int(self.bar_features.shape[1])
        self.n_ticks = int(self.tick_ask.shape[0])

        assert self.tick_bid.shape == self.tick_ask.shape
        assert self.tick_to_bar.shape[0] == self.n_ticks

    # --------------------- Helpers ---------------------
    def _pip_value_usd(self) -> float:
        # For EURUSD and similar USD-quoted pairs, exchange_rate ~ 1.0
        return ((self.lot * 100000.0) * self.pip_decimal) / max(1e-12, self.exchange_rate)

    def _first_valid_tick_after_warmup(self) -> int:
        """First tick whose mapped bar index has at least window_len-1 closed bars behind."""
        min_bar = self.window_len - 1
        idx = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        if idx >= self.n_ticks:
            raise ValueError("No valid ticks after warmup. Check data and warmup settings.")
        return idx

    # --------------------- Gym-like API ---------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng.seed(seed)

        if self.reset_balance_each_episode:
            self.balance = self.start_balance

        self.truncated = False
        self.done = False
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0

        self.t = self._first_valid_tick_after_warmup()
        self.current_tick_row = self.t
        self.episode_steps = 0

        obs = self._get_observation(self.t)
        return obs, {}

    def step(self, action):
        """
        Executes a single step:
        - HOLD: advance one tick, zero reward.
        - BUY/SELL: simulate until TP/SL or EOF; update PnL, balance, equity, drawdown; compute reward (pure).
        """
        # accept ints from the trainer
        if isinstance(action, (int, np.integer)):
            action = {0: "HOLD", 1: "BUY", 2: "SELL"}.get(int(action), "HOLD")

        if self.done:
            obs = self._get_observation(self.t)
            return obs, 0.0, True, self.truncated, {"note": "episode already done"}

        # episode step cap
        self.episode_steps = getattr(self, "episode_steps", 0) + 1
        if (self.max_steps_per_episode is not None) and (self.episode_steps >= self.max_steps_per_episode):
            self.done = True
            self.truncated = True
            obs = self._get_observation(self.t)
            return obs, 0.0, True, True, {"reason": "max_steps"}

        # guard: end of data
        if self.t >= self.n_ticks - 1:
            self.done = True
            obs = self._get_observation(self.t)
            return obs, 0.0, True, self.truncated, {"reason": "eof"}

        # HOLD action: walk one tick forward
        if action == "HOLD":
            self.t += 1
            self.current_tick_row = self.t
            obs = self._get_observation(self.t)
            return obs, 0.0, False, self.truncated, {"action": "HOLD"}

        # Entry prices at current tick
        entry_idx = self.t
        entry_ask = float(self.tick_ask[entry_idx])
        entry_bid = float(self.tick_bid[entry_idx])

        # Simulate BUY/SELL to TP/SL
        is_profitable = None
        tp_value = sl_value = None
        k = entry_idx

        if action == "BUY":
            tp_value = entry_ask + self.tp_pips * self.pip_decimal
            sl_value = entry_bid - self.sl_pips * self.pip_decimal
            # long exit at BID
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

        elif action == "SELL":
            tp_value = entry_bid - self.tp_pips * self.pip_decimal
            sl_value = entry_ask + self.sl_pips * self.pip_decimal
            # short exit at ASK
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

        else:
            # unknown action -> no-op advance (like HOLD)
            self.t += 1
            self.current_tick_row = self.t
            obs = self._get_observation(self.t)
            return obs, 0.0, False, self.truncated, {"action": action, "note": "unknown_action_noop"}

        # Move time to close tick
        self.t = k
        self.current_tick_row = self.t

        # EOF without TP/SL -> treat as SL
        if is_profitable is None:
            is_profitable = False
            if self.t >= self.n_ticks - 1:
                self.done = True

        # ---------- PnL ----------
        pip_value = self._pip_value_usd()
        profit = (pip_value * self.tp_pips) if is_profitable else -(pip_value * self.sl_pips)

        # Optional: subtract entry spread once
        if self.include_spread_cost:
            entry_spread = max(0.0, entry_ask - entry_bid)  # non-negative
            spread_cost_usd = entry_spread * (self.lot * 100000.0) / max(1e-12, self.exchange_rate)
            profit -= spread_cost_usd

        # Also report pips (signed)
        pnl_pips = float(self.tp_pips if is_profitable else -self.sl_pips)
        trade_duration_ticks = int(self.t - entry_idx)

        # ---------- State updates (single source of truth) ----------
        next_balance = self.balance + profit
        if next_balance <= 0.0:
            self.balance = 0.0
            self.equity = 0.0
            self.truncated = True
            self.done = True
        else:
            self.balance = next_balance
            self.equity = self.balance

        prev_max_dd = self.max_drawdown
        self.equity_peak = max(self.equity_peak, self.equity)
        dd_now = 0.0 if self.equity_peak == 0.0 else (self.equity_peak - self.equity) / self.equity_peak
        dd_now = max(0.0, dd_now)
        new_dd_increment = max(0.0, dd_now - prev_max_dd)
        self.max_drawdown = max(self.max_drawdown, dd_now)

        if self.max_drawdown >= self.max_dd_stop:
            self.done = True
            self.truncated = True

        # ---------- PURE reward ----------
        reward = self._calculate_reward(
            profit=profit, sl_pips=self.sl_pips, new_dd_increment=new_dd_increment
        )

        # ---------- Build outputs ----------
        obs = self._get_observation(self.t)
        info = {
            "action": action,
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
            "profit": float(profit),
            "profit_R": float(profit / max(1e-8, self._pip_value_usd() * self.sl_pips)),
            "pnl_pips": float(pnl_pips),
            "close_price": float(close_px),
            "trade_duration_ticks": int(trade_duration_ticks),
            "closed_trade": action in ("BUY", "SELL"),
        }
        return obs, float(reward), bool(self.done), bool(self.truncated), info

    # --------------------- Observations ---------------------
    def _get_observation(self, tick_idx: int):
        """
        Observation = [window_len * bar_feats] + [ask, bid, spread]
        (no position flag in per-trade/instant-close mode)
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

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        obs = np.hstack([feat_win.ravel(), np.array([ask, bid, spread], dtype=np.float32)]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------- Reward (pure) ---------------------
    def _calculate_reward(self, profit: float, sl_pips: float, new_dd_increment: float) -> float:
        """
        PURE reward: ±PnL in R units minus drawdown increment penalty.
        - denom = 1R = pip_value * SL_pips  (so TP/SL symmetric => ±1)
        - NO state changes here.
        """
        pip_value = self._pip_value_usd()
        denom = max(1e-8, pip_value * float(sl_pips))
        pnl_norm = profit / denom                # roughly ±1
        penalty = self.dd_penalty_lambda * float(new_dd_increment)
        reward = pnl_norm - penalty
        return float(reward)
