import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TradingEnvTick(gym.Env):
    """
    Tick-by-tick environment with persistent positions, NO auto TP/SL.

    - One step = advance by ONE tick.
    - Actions (Discrete(4)):
        0 = HOLD / NOOP
        1 = OPEN_LONG (if flat; otherwise NOOP)
        2 = OPEN_SHORT (if flat; otherwise NOOP)
        3 = CLOSE (if position open; otherwise NOOP)

    - Position marking:
        * Long MTM on BID, entry at ASK (spread realized immediately).
        * Short MTM on ASK, entry at BID.

    - Reward:
        * r_t = (equity_t - equity_{t-1}) / risk_per_trade_usd
        * Optional small time penalty while holding.
        * Optional drawdown penalty and/or early stop.

    - NO TP/SL:
        * The agent must explicitly CLOSE to realize PnL.
        * TP/SL kwargs are accepted but ignored (backward compatibility).

    - Episode termination:
        * End of data, max_steps_per_episode, equity <= 0, or max drawdown stop.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        pip_decimal: float,
        candles_file: str,
        tick_file: str,
        cache_dir: str = "cache_fx_EURUSD_D1",
        window_len: int = 32,
        # Deprecated (ignored) but kept for compatibility:
        tp_pips: float | None = None,
        sl_pips: float | None = None,
        lot: float = 1.0,
        risk_per_trade_usd: float = 1000.0,
        dd_penalty_lambda: float = 0.0,
        max_dd_stop: float = 0.30,
        max_steps_per_episode: int = 10_000,
        normalize_prices: bool = True,
        time_penalty_per_step: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.pip_decimal = float(pip_decimal)
        self.window_len = int(window_len)
        # tp/sl intentionally unused
        self.lot = float(lot)
        self.risk_per_trade_usd = float(risk_per_trade_usd)
        self.dd_penalty_lambda = float(dd_penalty_lambda)
        self.max_dd_stop = float(max_dd_stop)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.normalize_prices = bool(normalize_prices)
        self.time_penalty_per_step = float(time_penalty_per_step)

        self.cache_dir = cache_dir
        self._load_cache(cache_dir)

        # obs = bars window + [ask,bid,spread, pos_flag, pos_dir, entry_price_norm, time_in_pos_norm, unrealized_pnl_pips]
        self.extra_feats = 8
        self.obs_dim = self.window_len * self.n_feats + self.extra_feats
        high = np.full((self.obs_dim,), 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # RNG
        self.np_rng = np.random.default_rng(seed if seed is not None else 0)

        # Normalization anchors
        self.ask_mean = self.ask_std = None
        self.bid_mean = self.bid_std = None
        self.spread_mean = self.spread_std = None

        # Account state
        self.initial_balance = 100000.0
        self._reset_account()

        # Position state
        self._flat_position()

    # -------------------- Core API --------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

        self._reset_account()
        self._flat_position()

        # pick starting tick after warmup
        min_bar = self.window_len - 1
        base = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        hi = max(base + 1, self.n_ticks - self.max_steps_per_episode - 2)
        self.t = int(self.np_rng.integers(low=base, high=hi, endpoint=False))
        self.prev_equity = self._mark_to_equity()  # for delta-reward

        # fit normalizers on a window behind t
        lookback = max(0, self.t - 20000)
        ask_slice = np.asarray(self.tick_ask[lookback:self.t+1], dtype=np.float32)
        bid_slice = np.asarray(self.tick_bid[lookback:self.t+1], dtype=np.float32)
        spread_slice = ask_slice - bid_slice
        self.ask_mean, self.ask_std = float(ask_slice.mean()), float(ask_slice.std() + 1e-8)
        self.bid_mean, self.bid_std = float(bid_slice.mean()), float(bid_slice.std() + 1e-8)
        self.spread_mean, self.spread_std = float(spread_slice.mean()), float(spread_slice.std() + 1e-8)

        self.steps = 0
        self.terminated = False
        self.truncated = False

        return self._get_observation(self.t), {}

    def step(self, action: int):
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, self.terminated, self.truncated, {"note": "episode ended"}

        # apply action
        if action == 1:   # OPEN_LONG
            if not self.position_open:
                self._open_position(direction=1)
        elif action == 2: # OPEN_SHORT
            if not self.position_open:
                self._open_position(direction=-1)
        elif action == 3: # CLOSE
            if self.position_open:
                self._close_position(mtm_tick=self.t)

        # advance one tick
        if self.t >= self.n_ticks - 2:
            self.terminated = True
        else:
            self.t += 1

        # NO auto TP/SL here (agent must close)

        # reward = delta equity / denom - optional dd penalty - (time penalty if holding)
        current_equity = self._mark_to_equity()
        pnl_delta = current_equity - self.prev_equity
        self.prev_equity = current_equity

        denom = max(1e-8, self.risk_per_trade_usd)
        reward = pnl_delta / denom

        if self.dd_penalty_lambda > 0.0:
            self._update_drawdown(current_equity)
            if self.max_drawdown >= self.max_dd_stop:
                self.terminated = True
                self.truncated = True
            reward -= self.dd_penalty_lambda * self.new_dd_increment

        if self.time_penalty_per_step != 0.0 and self.position_open:
            reward -= self.time_penalty_per_step

        self.steps += 1
        if self.steps >= self.max_steps_per_episode:
            self.truncated = True

        info = {
            "position_open": self.position_open,
            "pos_dir": self.pos_dir,
            "entry_price": self.entry_price if self.position_open else None,
            # kept for compatibility; always None in this no-TP/SL variant
            "tp_price": None,
            "sl_price": None,
            "equity": current_equity,
            "balance": self.balance,
            "unrealized_pnl_pips": self._unrealized_pnl_pips(self.t) if self.position_open else 0.0,
            "max_drawdown": self.max_drawdown,
        }

        return self._get_observation(self.t), float(reward), self.terminated, self.truncated, info

    # -------------------- Position & PnL --------------------
    def _open_position(self, direction: int):
        self.position_open = True
        self.pos_dir = 1 if direction > 0 else -1
        ask = float(self.tick_ask[self.t])
        bid = float(self.tick_bid[self.t])
        self.entry_price = ask if self.pos_dir > 0 else bid
        self.entry_tick = self.t
        # keep attributes for compatibility, but unused
        self.tp_price = None
        self.sl_price = None

    def _close_position(self, mtm_tick: int):
        pnl = self._unrealized_pnl_usd(mtm_tick)
        self.balance = max(0.0, self.balance + pnl)
        self.position_open = False
        self.pos_dir = 0
        self.entry_price = None
        self.entry_tick = None
        self.tp_price = None
        self.sl_price = None
        self._update_drawdown(self.balance)

    def _unrealized_pnl_pips(self, tick_idx: int) -> float:
        if not self.position_open:
            return 0.0
        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        if self.pos_dir > 0:
            return (bid - self.entry_price) / self.pip_decimal
        else:
            return (self.entry_price - ask) / self.pip_decimal

    def _unrealized_pnl_usd(self, tick_idx: int) -> float:
        pip_value = ((self.lot * 100000) * self.pip_decimal) / 1.0
        return pip_value * self._unrealized_pnl_pips(tick_idx)

    def _mark_to_equity(self) -> float:
        if self.position_open:
            return self.balance + self._unrealized_pnl_usd(self.t)
        return self.balance

    # -------------------- Observation --------------------
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

        a = self._norm(ask, self.ask_mean, self.ask_std)
        b = self._norm(bid, self.bid_mean, self.bid_std)
        s = self._norm(spread, self.spread_mean, self.spread_std)

        pos_flag = 1.0 if self.position_open else 0.0
        pos_dir = float(self.pos_dir)
        entry_norm = 0.0 if self.entry_price is None else self._norm(
            self.entry_price,
            self.ask_mean if self.pos_dir>0 else self.bid_mean,
            self.ask_std  if self.pos_dir>0 else self.bid_std
        )
        time_in_pos = 0.0 if self.entry_tick is None else min(1.0, (tick_idx - self.entry_tick) / 5000.0)
        upnl_pips = self._unrealized_pnl_pips(tick_idx) if self.position_open else 0.0

        obs = np.hstack([
            feat_win.ravel(),
            np.array([a, b, s, pos_flag, pos_dir, entry_norm, time_in_pos, upnl_pips], dtype=np.float32)
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------- Helpers --------------------
    def _norm(self, x, m, s):
        if not self.normalize_prices or m is None or s is None or s == 0.0:
            return float(x)
        return float((x - m) / (s + 1e-8))

    def _update_drawdown(self, equity_now: float):
        if equity_now > self.equity_peak:
            self.equity_peak = equity_now
        dd_now = max(0.0, (self.equity_peak - equity_now) / (self.equity_peak + 1e-9))
        self.new_dd_increment = max(0.0, dd_now - self.max_drawdown)
        if dd_now > self.max_drawdown:
            self.max_drawdown = dd_now

    def _reset_account(self):
        self.balance = float(self.initial_balance)
        self.equity_peak = self.balance
        self.max_drawdown = 0.0
        self.new_dd_increment = 0.0

    def _flat_position(self):
        self.position_open = False
        self.pos_dir = 0
        self.entry_price = None
        self.entry_tick = None
        self.tp_price = None
        self.sl_price = None

    # -------------------- Data --------------------
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
