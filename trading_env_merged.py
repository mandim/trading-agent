import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# try:
#     from server import ZMQRepServer
# except Exception:
#     ZMQRepServer = None


class TradingEnv(gym.Env):
    """
    DQN-ready trading environment with realistic broker costs.

    Design:
      - Gymnasium API:
            reset(*, seed, options) -> (obs, info)
            step(action) -> (obs, reward, terminated, truncated, info)
      - Discrete actions: 0, 1, 2 interpreted based on position state:

            If flat (position_side == 0):
                0 -> WAIT        (stay flat)
                1 -> OPEN_LONG
                2 -> OPEN_SHORT

            If in a position (position_side != 0):
                0 -> HOLD        (keep position)
                1 -> CLOSE       (close at market, only if min_hold_bars satisfied)
                2 -> REVERSE     (optional; disabled by default)

      - One decision per BAR OPEN; intrabar tick simulation for TP/SL.
      - Train/eval split can be fraction-based or date-based.
      - Reward is realized-only on trade close (in R units), with optional trade penalty.

    Key choices:
      * We use tick BID/ASK directly; spread is implicit in those prices.
      * No explicit "spread_once" pips; this avoids double-counting.
      * Lot size is fixed (no percent-risk sizing to keep dynamics simple).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pip_decimal: float,
        candles_file: str,
        tick_file: str,
        cache_dir: str,

        # Trading config
        tp_pips: float = 50.0,
        sl_pips: float = 50.0,
        lot: float = 1.0,
        start_balance: float = 100_000.0,
        reset_balance_each_episode: bool = True,
        exchange_rate: float = 1.0,   # 1.0 for USD-quoted symbols like EURUSD

        # Broker cost config
        commission_per_lot_per_side_usd: float = 0.0,
        enable_commission: bool = False,
        enable_swaps: bool = False,
        swap_long_pips_per_day: float = 0.0,
        swap_short_pips_per_day: float = 0.0,
        slippage_pips_open: float = 0.0,
        slippage_pips_close: float = 0.0,
        slippage_mode: str = "fixed",   # "fixed", "uniform", "normal"
        enable_slippage: bool = True,
        slippage_pips: float = 0.0,   # simple fixed slippage in pips (fallback)
        other_fixed_cost_per_trade_usd: float = 0.0,
        include_spread_cost: bool = False,   # kept for backward compatibility, not used
        position_spreads_once_pips: float = 0.0,  # kept for backward compatibility, not used

        # Risk & penalties
        dd_penalty_lambda: float = 0.2,
        flat_penalty_R: float = 0.01,
        max_dd_stop: float = 0.30,
        trade_penalty_R: float = 0.2,
        min_hold_bars: int = 3,
        allow_reverse: bool = False,
        cooldown_bars_after_close: int = 0,

        # Episode / data split
        max_steps_per_episode: int | None = 5000,
        train_fraction: float = 0.7,
        eval_mode: bool = False,

        # Date-based split (preferred). If eval_start_date is set, it overrides train_fraction.
        eval_start_date: str | None = None,   # e.g. "2023-01-01"
        eval_end_date: str | None = None,     # e.g. "2025-12-31" (optional)
        train_start_date: str | None = None,  # e.g. "2019-01-01" (optional, for scaler fit + train start clamp)

        # Observation / normalization
        window_len: int = 32,
        normalize_prices: bool = True,
        normalize_bars: bool = True,

        # Runtime / integration
        start_server: bool = False,
        bind_address: str = "tcp://*:5555",
        seed: int | None = None,
    ):
        super().__init__()

        # Inputs kept only for reference (candles_file/tick_file may be unused)
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.cache_dir = cache_dir

        # Trading config
        self.pip_decimal = float(pip_decimal)
        self.pip_size = self.pip_decimal  # alias
        self.tp_pips = float(tp_pips)
        self.sl_pips = float(sl_pips)
        self.lot = float(lot)
        self.start_balance = float(start_balance)
        self.reset_balance_each_episode = bool(reset_balance_each_episode)
        self.exchange_rate = float(exchange_rate)

        # Broker cost config
        self.commission_per_lot_per_side_usd = float(commission_per_lot_per_side_usd)
        self.enable_commission = bool(enable_commission)
        self.enable_swaps = bool(enable_swaps)
        self.swap_long_pips_per_day = float(swap_long_pips_per_day)
        self.swap_short_pips_per_day = float(swap_short_pips_per_day)
        self.slippage_pips_open = float(slippage_pips_open)
        self.slippage_pips_close = float(slippage_pips_close)
        self.slippage_mode = slippage_mode
        self.enable_slippage = bool(enable_slippage)
        self.slippage_pips = float(slippage_pips)
        self.other_fixed_cost_per_trade_usd = float(other_fixed_cost_per_trade_usd)

        # kept for backward compatibility; spread is implicit in bid/ask
        self.include_spread_cost = bool(include_spread_cost)
        self.position_spreads_once_pips = float(position_spreads_once_pips)

        # Risk & penalties
        self.dd_penalty_lambda = float(dd_penalty_lambda)
        self.max_dd_stop = float(max_dd_stop)
        self.trade_penalty_R = float(trade_penalty_R)
        self.min_hold_bars = int(min_hold_bars)
        self.allow_reverse = bool(allow_reverse)
        self.cooldown_bars_after_close = int(cooldown_bars_after_close)
        self.cooldown_remaining = 0
        self.flat_penalty_R = float(flat_penalty_R)

        # Episode / split
        self.max_steps_per_episode = int(max_steps_per_episode) if max_steps_per_episode is not None else None
        self.train_fraction = float(train_fraction)
        self.eval_mode = bool(eval_mode)

        self.eval_start_date = eval_start_date
        self.eval_end_date = eval_end_date
        self.train_start_date = train_start_date

        # Observation config
        self.window_len = int(window_len)
        self.normalize_prices = bool(normalize_prices)
        self.normalize_bars = bool(normalize_bars)

        # RNG
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        # Load cached data (also computes split indices + bars scaler)
        self._load_cache(self.cache_dir)

        # Observation: window_len * n_feats + (price_feats=3) + (pos_feats=3)
        self.extra_obs_dim = 6
        self.obs_dim = self.window_len * self.n_feats + self.extra_obs_dim

        high = np.full((self.obs_dim,), 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Account / episode state
        self.balance = self.start_balance
        self.equity = self.start_balance
        self.equity_peak = self.start_balance
        self.max_drawdown = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0
        self.t = 0
        self.current_tick_row = 0

        # Cost trackers
        self.cumulative_commission = 0.0
        self.cumulative_swap = 0.0
        self.cumulative_other_costs = 0.0

        # Position state
        self.position_side = 0  # 0 flat, +1 long, -1 short
        self.position_entry_idx = None
        self.position_entry_ask = None
        self.position_entry_bid = None
        self.position_tp = None
        self.position_sl = None
        self.position_is_long = None
        self.position_duration_ticks = 0

        # Normalization anchors (price)
        self.ask_mean = self.ask_std = None
        self.bid_mean = self.bid_std = None
        self.spread_mean = self.spread_std = None

        # Optional ZMQ server placeholder (avoid attribute errors)
        self.server = None
        # if start_server and ZMQRepServer is not None:
        #     self.server = ZMQRepServer(bind_address, self._handle_request)

    # -------------------------------------------------------------------------
    # Optional server
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
        if not isinstance(request, dict):
            return {"reply": "invalid_request"}

        cmd = str(request.get("cmd", "")).upper().strip()

        if cmd == "HOLD":
            action = 0
        elif cmd == "CLOSE":
            action = 1 if self.position_side != 0 else 0
        elif cmd == "BUY":
            if self.position_side == 0:
                action = 1
            elif self.position_side < 0:
                action = 2
            else:
                action = 0
        elif cmd == "SELL":
            if self.position_side == 0:
                action = 2
            elif self.position_side > 0:
                action = 2
            else:
                action = 0
        elif cmd == "REVERSE":
            action = 2
        else:
            return {"reply": "unknown_command", "received": request}

        obs, reward, terminated, truncated, info = self.step(action)
        return {
            "obs": obs.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }

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

        # Reset account state
        self.equity = float(self.balance)
        self.equity_peak = float(self.balance)
        self.max_drawdown = 0.0
        self.terminated = False
        self.truncated = False
        self.steps = 0

        # Reset cost trackers
        self.cumulative_commission = 0.0
        self.cumulative_swap = 0.0
        self.cumulative_other_costs = 0.0

        # Reset position
        self.position_side = 0
        self.position_entry_idx = None
        self.position_entry_ask = None
        self.position_entry_bid = None
        self.position_tp = None
        self.position_sl = None
        self.position_is_long = None
        self.position_duration_ticks = 0

        # Choose starting tick (train vs eval region)
        min_bar = self.window_len - 1
        base = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        if base >= self.n_ticks:
            raise ValueError("Warmup exceeds available ticks; check window_len or data.")

        # Enforce train_start_date for training mode (prevents sampling before requested start)
        if (not self.eval_mode) and (self.train_start_bar is not None) and (int(self.train_start_bar) > 0):
            train_start_tick = self._bar_to_tick_index(int(self.train_start_bar))
            base = max(base, int(train_start_tick))

        split_tick = int(self.split_tick) if self.split_tick is not None else int(self.n_ticks * self.train_fraction)

        if self.eval_mode:
            self.t = max(base, split_tick)
        else:
            start_low = base
            start_high = max(start_low + 1, split_tick)
            if start_low >= start_high:
                self.t = base
            else:
                self.t = int(self._rng.integers(low=start_low, high=start_high, endpoint=False))

        self.current_tick_row = self.t

        # Initialize price normalizers using history up to starting tick
        self._update_price_normalizers(self.t)

        obs = self._get_observation(self.t)
        return obs, {}

    def step(self, action: int):
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, True, True, {}

        # ---- helpers ----
        def is_bar_open(idx: int) -> bool:
            if idx <= 0:
                return True
            return int(self.tick_to_bar[idx]) != int(self.tick_to_bar[idx - 1])

        def advance_to_bar_open():
            while self.t < (self.n_ticks - 1) and not is_bar_open(self.t):
                self.t += 1

        def bar_of(idx: int) -> int:
            return int(self.tick_to_bar[min(max(idx, 0), self.n_ticks - 1)])

        # ---- 0) align to decision point ----
        advance_to_bar_open()

        # If eval_end_date is set, stop evaluation once we reach/past that bar.
        if self.eval_mode and (self.eval_end_bar is not None):
            cur_bar = int(self.tick_to_bar[self.t])
            if cur_bar >= int(self.eval_end_bar):
                self.terminated = True
                return self._get_observation(self.t), 0.0, True, False, {}

        if self.t >= self.n_ticks - 1:
            self.terminated = True
            return self._get_observation(self.t), 0.0, True, False, {}

        # cache for info / drawdown updates
        _ = float(self._equity_mtm())  # equity_prev not used in realized-only reward version

        # bookkeeping (per decision step)
        opened_trade = False
        closed_trade = False
        reversed_trade = False
        blocked_by_cooldown = False

        # Cooldown: prevent re-entry for N bars after any close (tp/sl/manual/eoe)
        if self.cooldown_remaining > 0 and self.position_side == 0 and action in (1, 2):
            blocked_by_cooldown = True
            action = 0  # force WAIT while flat

        # Decrement cooldown counter *after* potential blocking check (counts bars)
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        exit_reason = None

        trade_pnl_pips = 0.0
        trade_pnl_usd = 0.0
        trade_profit_R = 0.0
        is_profitable = False

        # For R normalization
        pip_value = float(self._pip_value_usd())
        one_R_usd = max(1e-9, pip_value * float(self.sl_pips))

        # current prices at bar open
        bid0 = float(self.tick_bid[self.t])
        ask0 = float(self.tick_ask[self.t])

        # compute how long we've held (bars)
        pos_age_bars = 0
        if self.position_side != 0 and self.position_entry_idx is not None:
            pos_age_bars = bar_of(self.t) - bar_of(self.position_entry_idx)

        can_manual_exit = (pos_age_bars >= int(self.min_hold_bars))

        # ---- commission/slippage helpers ----
        self._last_commission_usd = 0.0
        self._last_swap_usd = 0.0
        self._last_other_cost_usd = 0.0

        def apply_commission_side():
            if bool(self.enable_commission):
                c = float(self._commission_usd(round_turn=False))
                self.balance -= c
                self.cumulative_commission += c
                self._last_commission_usd += c

        def apply_slippage(price: float, is_entry: bool, is_long: bool) -> float:
            if not bool(self.enable_slippage):
                return price

            base_pips = float(self.slippage_pips_open if is_entry else self.slippage_pips_close)
            if base_pips <= 0.0:
                base_pips = float(self.slippage_pips)

            slip_pips = float(self._sample_slippage(base_pips))
            slip = slip_pips * float(self.pip_decimal)

            if is_entry:
                return price + slip if is_long else price - slip
            else:
                return price - slip if is_long else price + slip

        def close_position(reason: str):
            nonlocal trade_pnl_pips, trade_pnl_usd, trade_profit_R, is_profitable, exit_reason

            bid = float(self.tick_bid[self.t])
            ask = float(self.tick_ask[self.t])
            is_long = bool(self.position_is_long)

            close_bid = apply_slippage(bid, is_entry=False, is_long=True)    # long exits on bid
            close_ask = apply_slippage(ask, is_entry=False, is_long=False)   # short exits on ask

            trade_pnl_pips = float(self._pnl_from_prices(
                is_long=is_long,
                entry_ask=float(self.position_entry_ask),
                entry_bid=float(self.position_entry_bid),
                close_bid=float(close_bid),
                close_ask=float(close_ask),
            ))
            trade_pnl_usd = float(trade_pnl_pips) * pip_value
            self.balance += trade_pnl_usd

            # commission on close
            apply_commission_side()

            # swap + other fixed costs (charged on close)
            if self.position_entry_idx is not None:
                swap_usd = float(self._compute_swap_usd(
                    is_long=is_long,
                    entry_idx=int(self.position_entry_idx),
                    exit_idx=int(self.t),
                ))
            else:
                swap_usd = 0.0

            if bool(self.enable_swaps) and swap_usd != 0.0:
                self.balance -= swap_usd
                self.cumulative_swap += swap_usd
                self._last_swap_usd += swap_usd

            other_usd = float(self.other_fixed_cost_per_trade_usd) if float(self.other_fixed_cost_per_trade_usd) != 0.0 else 0.0
            if other_usd != 0.0:
                self.balance -= other_usd
                self.cumulative_other_costs += other_usd
                self._last_other_cost_usd += other_usd

            # clear position
            self.position_side = 0
            self.position_entry_idx = None
            self.position_entry_ask = None
            self.position_entry_bid = None
            self.position_tp = None
            self.position_sl = None
            self.position_is_long = None
            self.position_duration_ticks = 0

            exit_reason = reason
            trade_profit_R = float(trade_pnl_usd) / one_R_usd
            is_profitable = (trade_pnl_usd > 0.0)

        # ---- 1) apply action once (at bar open) ----
        if self.position_side == 0:
            if action == 1:
                opened_trade = True
                apply_commission_side()
                entry_ask = apply_slippage(ask0, is_entry=True, is_long=True)
                entry_bid = bid0
                self.position_side = 1
                self.position_entry_idx = self.t
                self.position_entry_ask = float(entry_ask)
                self.position_entry_bid = float(entry_bid)
                self.position_is_long = True
                self.position_tp = float(self.position_entry_ask + float(self.tp_pips) * float(self.pip_decimal))
                self.position_sl = float(self.position_entry_bid - float(self.sl_pips) * float(self.pip_decimal))
            elif action == 2:
                opened_trade = True
                apply_commission_side()
                entry_bid = apply_slippage(bid0, is_entry=True, is_long=False)
                entry_ask = ask0
                self.position_side = -1
                self.position_entry_idx = self.t
                self.position_entry_ask = float(entry_ask)
                self.position_entry_bid = float(entry_bid)
                self.position_is_long = False
                self.position_tp = float(self.position_entry_bid - float(self.tp_pips) * float(self.pip_decimal))
                self.position_sl = float(self.position_entry_ask + float(self.sl_pips) * float(self.pip_decimal))
        else:
            if action == 1 and can_manual_exit:
                closed_trade = True
                close_position("manual_close")
            elif action == 2 and bool(self.allow_reverse) and can_manual_exit:
                reversed_trade = True
                closed_trade = True
                prev_long = bool(self.position_is_long)
                close_position("reverse_close")

                # open opposite
                if prev_long:
                    opened_trade = True
                    apply_commission_side()
                    entry_bid = apply_slippage(bid0, is_entry=True, is_long=False)
                    entry_ask = ask0
                    self.position_side = -1
                    self.position_entry_idx = self.t
                    self.position_entry_ask = float(entry_ask)
                    self.position_entry_bid = float(entry_bid)
                    self.position_is_long = False
                    self.position_tp = float(self.position_entry_bid - float(self.tp_pips) * float(self.pip_decimal))
                    self.position_sl = float(self.position_entry_ask + float(self.sl_pips) * float(self.pip_decimal))
                else:
                    opened_trade = True
                    apply_commission_side()
                    entry_ask = apply_slippage(ask0, is_entry=True, is_long=True)
                    entry_bid = bid0
                    self.position_side = 1
                    self.position_entry_idx = self.t
                    self.position_entry_ask = float(entry_ask)
                    self.position_entry_bid = float(entry_bid)
                    self.position_is_long = True
                    self.position_tp = float(self.position_entry_ask + float(self.tp_pips) * float(self.pip_decimal))
                    self.position_sl = float(self.position_entry_bid - float(self.sl_pips) * float(self.pip_decimal))
            # else HOLD

        # ---- 2) simulate intrabar until next bar open ----
        cur_bar = bar_of(self.t)
        i = self.t
        while i < self.n_ticks - 1:
            if i > self.t and bar_of(i) != cur_bar:
                break
            self.t = i

            if self.position_side != 0:
                self.position_duration_ticks += 1

            if self.position_side != 0 and not closed_trade:
                bid = float(self.tick_bid[self.t])
                ask = float(self.tick_ask[self.t])
                if bool(self.position_is_long):
                    if bid >= float(self.position_tp):
                        closed_trade = True
                        close_position("tp")
                    elif bid <= float(self.position_sl):
                        closed_trade = True
                        close_position("sl")
                else:
                    if ask <= float(self.position_tp):
                        closed_trade = True
                        close_position("tp")
                    elif ask >= float(self.position_sl):
                        closed_trade = True
                        close_position("sl")

            i += 1

        # move to next decision point (next bar open)
        self.t = min(i, self.n_ticks - 1)
        advance_to_bar_open()

        # step counter / termination
        self.steps += 1
        if self.max_steps_per_episode is not None and self.steps >= int(self.max_steps_per_episode):
            self.truncated = True
        if self.t >= self.n_ticks - 1:
            self.terminated = True

        # forced close at end of episode
        if (self.terminated or self.truncated) and self.position_side != 0 and not closed_trade:
            closed_trade = True
            close_position("eoe_close")

        # update equity / drawdown stats
        equity_now = float(self._equity_mtm())
        self.equity = equity_now
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity

        dd = 0.0
        if self.equity_peak > 0:
            dd = (self.equity_peak - self.equity) / self.equity_peak
        if dd > self.max_drawdown:
            self.max_drawdown = dd
        if self.max_drawdown >= float(self.max_dd_stop):
            self.truncated = True

        # ---- 3) reward: realized only ----
        reward = 0.0
        if closed_trade:
            reward = float(trade_profit_R - float(self.trade_penalty_R))
        else:
            if self.position_side == 0 and action == 0 and float(self.flat_penalty_R) > 0:
                reward = -float(self.flat_penalty_R)

        obs = self._get_observation(self.t)

        # If we closed a trade this step, start cooldown from next bar
        if closed_trade and self.cooldown_bars_after_close > 0:
            self.cooldown_remaining = int(self.cooldown_bars_after_close)

        info = {
            "t": int(self.t),
            "bar_index": int(self.tick_to_bar[self.t]),
            "equity": float(self.equity),
            "balance": float(self.balance),
            "position_side": int(self.position_side),
            "max_drawdown": float(self.max_drawdown),

            "opened_trade": bool(opened_trade),
            "closed_trade": bool(closed_trade),
            "reversed_trade": bool(reversed_trade),
            "exit_reason": exit_reason,

            "profit": float(trade_pnl_usd),
            "profit_R": float(trade_profit_R),
            "pnl_pips": float(trade_pnl_pips),
            "is_profitable": bool(is_profitable),

            "commission_usd": float(getattr(self, "_last_commission_usd", 0.0)),
            "swap_usd": float(getattr(self, "_last_swap_usd", 0.0)),
            "other_cost_usd": float(getattr(self, "_last_other_cost_usd", 0.0)),

            # NEW: expose cooldown diagnostics (for TB + debugging)
            "blocked_by_cooldown": bool(blocked_by_cooldown),
            "cooldown_remaining": int(self.cooldown_remaining),
        }
        return obs, float(reward), bool(self.terminated), bool(self.truncated), info

    # -------------------------------------------------------------------------
    # Normalization / sampling helpers
    # -------------------------------------------------------------------------
    def _update_price_normalizers(self, tick_idx: int):
        if not self.normalize_prices:
            return

        if tick_idx <= 0:
            window_ask = self.tick_ask[:1].astype(np.float32)
            window_bid = self.tick_bid[:1].astype(np.float32)
        else:
            hi = min(self.n_ticks, int(tick_idx) + 1)
            window_ask = self.tick_ask[:hi].astype(np.float32)
            window_bid = self.tick_bid[:hi].astype(np.float32)

        window_spread = (window_ask - window_bid).astype(np.float32)

        self.ask_mean = float(window_ask.mean())
        self.ask_std = float(window_ask.std() + 1e-8)

        self.bid_mean = float(window_bid.mean())
        self.bid_std = float(window_bid.std() + 1e-8)

        self.spread_mean = float(window_spread.mean())
        self.spread_std = float(window_spread.std() + 1e-8)

    def _sample_slippage(self, base_pips: float) -> float:
        if base_pips <= 0:
            return 0.0

        mode = (self.slippage_mode or "fixed").lower()
        if mode == "fixed":
            return base_pips
        if mode == "uniform":
            return float(self._rng.uniform(0.0, base_pips))
        if mode == "normal":
            return float(abs(self._rng.normal(loc=0.0, scale=base_pips)))
        return base_pips

    def _pip_value_usd_per_lot(self) -> float:
        return (100000.0 * self.pip_decimal) / max(1e-12, self.exchange_rate)

    def _pip_value_usd(self) -> float:
        return self._pip_value_usd_per_lot() * self.lot

    def _commission_usd(self, round_turn: bool = False) -> float:
        if not self.enable_commission or self.commission_per_lot_per_side_usd <= 0.0:
            return 0.0
        sides = 2 if round_turn else 1
        return float(self.lot * self.commission_per_lot_per_side_usd * sides)

    def _pnl_from_prices(self, is_long: bool, entry_ask: float, entry_bid: float, close_bid: float, close_ask: float) -> float:
        if is_long:
            return (close_bid - entry_ask) / self.pip_decimal
        else:
            return (entry_bid - close_ask) / self.pip_decimal

    def _compute_swap_usd(self, is_long: bool, entry_idx: int, exit_idx: int) -> float:
        if not self.enable_swaps or exit_idx <= entry_idx:
            return 0.0
        if not hasattr(self, "bar_times") or self.bar_times is None:
            return 0.0

        entry_bar = int(self.tick_to_bar[entry_idx])
        exit_bar = int(self.tick_to_bar[exit_idx])
        if exit_bar <= entry_bar:
            return 0.0

        t0 = self.bar_times[entry_bar]
        t1 = self.bar_times[exit_bar]

        try:
            if np.issubdtype(self.bar_times.dtype, np.datetime64):
                days_held = float((t1 - t0) / np.timedelta64(1, "D"))
            else:
                days_held = float(t1 - t0) / (24.0 * 60.0 * 60.0)
        except Exception:
            days_held = float(exit_bar - entry_bar)

        days_held = max(0.0, days_held)
        if days_held <= 0.0:
            return 0.0

        rate = self.swap_long_pips_per_day if is_long else self.swap_short_pips_per_day
        if rate == 0.0:
            return 0.0

        return float(rate * self._pip_value_usd() * days_held)

    def _equity_mtm(self) -> float:
        eq = float(self.balance)

        if self.position_side == 0 or self.position_entry_idx is None:
            return eq

        idx = min(max(self.t, 0), self.n_ticks - 1)
        bid = float(self.tick_bid[idx])
        ask = float(self.tick_ask[idx])
        is_long = bool(self.position_is_long)

        base_pips = self._pnl_from_prices(
            is_long,
            self.position_entry_ask,
            self.position_entry_bid,
            bid,
            ask,
        )

        pip_value = self._pip_value_usd()
        float_profit_usd = float(base_pips * pip_value)
        return float(eq + float_profit_usd)

    def _price_norm(self, x: float, mean: float, std: float) -> float:
        if (not self.normalize_prices) or mean is None or std is None or std == 0.0:
            return float(x)
        return float((x - mean) / (std + 1e-8))

    def _get_obs(self):
        try:
            self._update_price_normalizers(self.t)
        except Exception:
            pass
        return self._get_observation(self.t)

    def _get_observation(self, tick_idx: int):
        bar_idx = int(self.tick_to_bar[tick_idx])
        L = self.window_len
        start = bar_idx - L + 1

        if start < 0:
            pad = np.zeros((-start, self.n_feats), dtype=np.float32)
            win = np.asarray(self.bar_features[0: bar_idx + 1], dtype=np.float32)
            feat_win = np.vstack([pad, win])
        else:
            feat_win = np.asarray(self.bar_features[start: bar_idx + 1], dtype=np.float32)

        if self.normalize_bars and hasattr(self, "bars_mean") and self.bars_mean is not None:
            feat_win = (feat_win - self.bars_mean) / (self.bars_std + 1e-8)

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        a = self._price_norm(ask, self.ask_mean, self.ask_std)
        b = self._price_norm(bid, self.bid_mean, self.bid_std)
        s = self._price_norm(spread, self.spread_mean, self.spread_std)
        price_feats = np.array([a, b, s], dtype=np.float32)

        if self.position_side == 0 or self.position_entry_idx is None:
            pos_dir = 0.0
            pos_pnl_R = 0.0
            pos_age = 0.0
        else:
            is_long = bool(self.position_is_long)
            base_pips = self._pnl_from_prices(
                is_long,
                self.position_entry_ask,
                self.position_entry_bid,
                bid,
                ask,
            )
            pip_value = self._pip_value_usd()
            float_profit_usd = float(base_pips * pip_value)
            one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
            pos_pnl_R = float(float_profit_usd / one_R_usd)
            pos_dir = 1.0 if is_long else -1.0

            # bars-based age
            entry_bar = int(self.tick_to_bar[self.position_entry_idx])
            pos_age_bars = max(0, bar_idx - entry_bar)
            pos_age = float(pos_age_bars) / 1000.0

        pos_feats = np.array([pos_dir, pos_pnl_R, pos_age], dtype=np.float32)

        obs = np.hstack([feat_win.ravel(), price_feats, pos_feats]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------------------
    # Date-based split helpers
    # -------------------------------------------------------------------------
    def _date_to_bar_index(self, date_str: str) -> int:
        if date_str is None:
            return 0

        target_dt = np.datetime64(date_str, "D")
        bt = self.bar_times

        if np.issubdtype(bt.dtype, np.datetime64):
            bt_d = bt.astype("datetime64[D]")
            idx = int(np.searchsorted(bt_d, target_dt, side="left"))
            return int(np.clip(idx, 0, self.n_bars))
        else:
            epoch = np.datetime64("1970-01-01", "D")
            days = (target_dt - epoch).astype(np.int64)
            target_sec = float(days) * 86400.0
            idx = int(np.searchsorted(bt.astype(np.float64), target_sec, side="left"))
            return int(np.clip(idx, 0, self.n_bars))

    def _bar_to_tick_index(self, bar_idx: int) -> int:
        bar_idx = int(np.clip(bar_idx, 0, self.n_bars))
        return int(np.searchsorted(self.tick_to_bar, bar_idx, side="left"))

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

        # ---------------- Date-based split (preferred) ----------------
        self.split_bar = None
        self.split_tick = None
        self.eval_end_bar = None
        self.eval_end_tick = None
        self.train_start_bar = 0

        if self.train_start_date:
            self.train_start_bar = self._date_to_bar_index(self.train_start_date)

        if self.eval_start_date:
            self.split_bar = self._date_to_bar_index(self.eval_start_date)
            self.split_tick = self._bar_to_tick_index(self.split_bar)

            if self.eval_end_date:
                self.eval_end_bar = self._date_to_bar_index(self.eval_end_date)
                self.eval_end_tick = self._bar_to_tick_index(self.eval_end_bar)
        else:
            # fallback to fraction-based split
            self.split_bar = int(self.n_bars * float(self.train_fraction))
            self.split_tick = int(self.n_ticks * float(self.train_fraction))

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
                train_lo = int(getattr(self, "train_start_bar", 0))
                train_hi = int(getattr(self, "split_bar", int(self.n_bars * float(self.train_fraction))))
                train_lo = max(0, min(train_lo, self.n_bars))
                train_hi = max(train_lo + 1, min(train_hi, self.n_bars))

                train_feats = np.asarray(self.bar_features[train_lo:train_hi], dtype=np.float32)
                self.bars_mean = train_feats.mean(axis=0)
                self.bars_std = train_feats.std(axis=0) + 1e-8
