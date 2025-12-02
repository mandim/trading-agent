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
    DQN-ready trading environment with realistic broker costs.

    Design goals:
      - Gymnasium API:
            reset(*, seed, options) -> (obs, info)
            step(action) -> (obs, reward, terminated, truncated, info)
      - Discrete actions: 0=HOLD, 1=BUY, 2=SELL
      - Tick-driven TP/SL exits
      - Train / eval split with random starting ticks
      - Optional price & bar normalization
      - Simple, well-scaled reward:
            reward = profit_usd / reward_scale_usd - λ * ΔDD
      - Mark-to-market equity, used for drawdown and EOF closing
      - Optional ZeroMQ server for MT4 integration

    Key choices:
      * We use the tick BID/ASK directly. Spread is implicit in those prices.
      * No extra "spread_once" pips are subtracted; this avoids double-counting.
      * Lot size is fixed (no percent-risk sizing to keep dynamics simple).
    """

    metadata = {"render_modes": []}

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------
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
        other_fixed_cost_per_trade_usd: float = 0.0,

        # Risk & penalties
        dd_penalty_lambda: float = 0.5,
        max_dd_stop: float = 0.30,

        # Episode / data split
        max_steps_per_episode: int | None = 5000,
        train_fraction: float = 0.7,
        eval_mode: bool = False,

        # Reward scaling
        reward_scale_usd: float = 100.0,   # profit_usd / reward_scale_usd

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
        self.other_fixed_cost_per_trade_usd = float(other_fixed_cost_per_trade_usd)

        # Risk & penalties
        self.dd_penalty_lambda = float(dd_penalty_lambda)
        self.max_dd_stop = float(max_dd_stop)

        # Episode / split
        self.max_steps_per_episode = int(max_steps_per_episode) if max_steps_per_episode is not None else None
        self.train_fraction = float(train_fraction)
        self.eval_mode = bool(eval_mode)

        # Reward
        self.reward_scale_usd = float(reward_scale_usd)

        # Observation config
        self.window_len = int(window_len)
        self.normalize_prices = bool(normalize_prices)
        self.normalize_bars = bool(normalize_bars)

        # RNG
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        # Load cached data
        self._load_cache(self.cache_dir)  # sets: bar_features, tick_ask, tick_bid, tick_to_bar, etc.

        # Gym spaces:
        #   - window_len * n_feats from bar_features
        #   - 3 price features  : [ask_norm, bid_norm, spread_norm]
        #   - 3 position features: [pos_dir, pos_pnl_R, pos_age]
        self.extra_obs_dim = 3 + 3
        self.obs_dim = self.window_len * self.n_feats + self.extra_obs_dim

        high = np.full((self.obs_dim,), 1e6, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0=HOLD,1=BUY,2=SELL

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
        # 0 = flat, +1 = long, -1 = short
        self.position_side = 0
        self.position_entry_idx = None
        self.position_entry_ask = None
        self.position_entry_bid = None
        self.position_tp = None
        self.position_sl = None
        self.position_is_long = None
        self.position_duration_ticks = 0

        # Normalization anchors
        self.ask_mean = self.ask_std = None
        self.bid_mean = self.bid_std = None
        self.spread_mean = self.spread_std = None

        # Optional ZMQ server
        self.server = None
        if start_server and ZMQRepServer is not None:
            self.server = ZMQRepServer(bind_address, self._handle_request)

    # -------------------------------------------------------------------------
    # ZeroMQ (optional)
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

        split_tick = int(self.n_ticks * self.train_fraction)

        if self.eval_mode:
            # Deterministic MT4-style evaluation on last 30% region
            self.t = max(base, split_tick)
        else:
            # Training: random start in [base, split_tick)
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
        # If episode already ended, no-op
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, self.terminated, self.truncated, {
                "note": "episode already ended"
            }

        # End-of-data guard: close open trade MTM and end
        if self.t >= self.n_ticks - 1:
            reward, info = self._handle_eof()
            obs = self._get_observation(self.n_ticks - 1)
            return obs, float(reward), True, bool(self.truncated), info

        # Determine if this tick is first of its bar (for eval bar-open gating)
        cur_bar = int(self.tick_to_bar[self.t])
        prev_bar = int(self.tick_to_bar[self.t - 1]) if self.t > 0 else -1
        is_new_bar = (cur_bar != prev_bar)

        # Update price normalizers on each new bar
        if is_new_bar:
            self._update_price_normalizers(self.t)

        # In eval_mode, only act on bar opens
        if self.eval_mode and not is_new_bar:
            action = 0  # force HOLD

        # Map action code
        if action == 0:
            act = "HOLD"
        elif action == 1:
            act = "BUY"
        elif action == 2:
            act = "SELL"
        else:
            act = "HOLD"

        # Current prices
        bid = float(self.tick_bid[self.t])
        ask = float(self.tick_ask[self.t])

        closed_trade = False
        net_profit_usd = 0.0
        gross_profit_usd = 0.0
        pnl_pips = 0.0
        commission_usd = 0.0
        swap_usd = 0.0
        other_cost_usd = 0.0
        profit_R = 0.0  # keep for logging only; 1R = sl_pips * pip_value
        trade_duration_ticks = 0
        dd_now = float(self.max_drawdown)
        new_dd_increment = 0.0

        # =====================================================================
        # 1) Check TP/SL for existing position
        # =====================================================================
        if self.position_side != 0 and self.position_entry_idx is not None:
            is_long = bool(self.position_is_long)
            self.position_duration_ticks += 1

            exit_reason = None

            if is_long:
                # long -> TP/SL evaluated on BID
                if self.position_tp is not None and bid >= self.position_tp:
                    exit_reason = "tp"
                elif self.position_sl is not None and bid <= self.position_sl:
                    exit_reason = "sl"
            else:
                # short -> TP/SL evaluated on ASK
                if self.position_tp is not None and ask <= self.position_tp:
                    exit_reason = "tp"
                elif self.position_sl is not None and ask >= self.position_sl:
                    exit_reason = "sl"

            if exit_reason is not None:
                exit_idx = self.t

                # Closing price base (before slippage)
                if is_long:
                    close_px_bid = bid
                    close_px_ask = ask
                else:
                    close_px_bid = bid
                    close_px_ask = ask

                # Slippage on close, if any
                if self.slippage_pips_close > 0.0:
                    slip = self._sample_slippage(self.slippage_pips_close) * self.pip_decimal
                    if is_long:
                        close_px_bid -= slip  # worse price when closing long
                    else:
                        close_px_ask += slip  # worse price when closing short

                # Base PnL in pips
                if exit_reason == "tp":
                    base_pips = self.tp_pips
                elif exit_reason == "sl":
                    base_pips = -self.sl_pips
                else:
                    base_pips = self._pnl_from_prices(
                        is_long,
                        self.position_entry_ask,
                        self.position_entry_bid,
                        close_px_bid,
                        close_px_ask,
                    )

                pnl_pips = float(base_pips)
                pip_value = self._pip_value_usd()
                gross_profit_usd = float(pnl_pips * pip_value)
                trade_duration_ticks = int(self.position_duration_ticks)

                # Swaps
                swap_usd = self._compute_swap_usd(is_long, self.position_entry_idx, exit_idx)

                # Commission & other costs
                commission_usd = self._commission_usd(round_turn=True)
                other_cost_usd = float(self.other_fixed_cost_per_trade_usd)

                # Net PnL
                net_profit_usd = gross_profit_usd + swap_usd - commission_usd - other_cost_usd

                # Track cumulative costs
                self.cumulative_commission += commission_usd
                self.cumulative_swap += swap_usd
                self.cumulative_other_costs += other_cost_usd

                # Update balance & equity, drawdown
                next_balance = self.balance + net_profit_usd
                if next_balance <= 0.0:
                    self.balance = 0.0
                    self.equity = 0.0
                    self.terminated = True
                    self.truncated = True
                else:
                    self.balance = float(next_balance)
                    self.equity = self._equity_mtm()

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

                # 1R reference for logging
                one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
                profit_R = float(net_profit_usd / one_R_usd)

                closed_trade = True

                # Clear position state
                self.position_side = 0
                self.position_entry_idx = None
                self.position_entry_ask = None
                self.position_entry_bid = None
                self.position_tp = None
                self.position_sl = None
                self.position_is_long = None
                self.position_duration_ticks = 0

        # =====================================================================
        # 2) Process action (open new trade only if flat)
        # =====================================================================
        if self.position_side == 0 and not self.terminated and not self.truncated:
            if act == "BUY":
                entry_idx = self.t
                entry_ask = ask
                entry_bid = bid

                # Slippage on open
                if self.slippage_pips_open > 0.0:
                    slip = self._sample_slippage(self.slippage_pips_open) * self.pip_decimal
                    entry_ask += slip  # pay more for long entry

                tp_value = entry_ask + self.tp_pips * self.pip_decimal
                sl_value = entry_bid - self.sl_pips * self.pip_decimal

                self.position_side = 1
                self.position_entry_idx = entry_idx
                self.position_entry_ask = entry_ask
                self.position_entry_bid = entry_bid
                self.position_tp = tp_value
                self.position_sl = sl_value
                self.position_is_long = True
                self.position_duration_ticks = 0

            elif act == "SELL":
                entry_idx = self.t
                entry_ask = ask
                entry_bid = bid

                # Slippage on open
                if self.slippage_pips_open > 0.0:
                    slip = self._sample_slippage(self.slippage_pips_open) * self.pip_decimal
                    entry_bid -= slip  # receive less on short entry

                tp_value = entry_bid - self.tp_pips * self.pip_decimal
                sl_value = entry_ask + self.sl_pips * self.pip_decimal

                self.position_side = -1
                self.position_entry_idx = entry_idx
                self.position_entry_ask = entry_ask
                self.position_entry_bid = entry_bid
                self.position_tp = tp_value
                self.position_sl = sl_value
                self.position_is_long = False
                self.position_duration_ticks = 0
            # HOLD when flat: do nothing

                # =====================================================================
        
        # 3) Reward & bookkeeping (including step-limit truncation)
        # =====================================================================
        self.steps += 1
        hit_step_limit = (
            self.max_steps_per_episode is not None
            and self.steps >= self.max_steps_per_episode
        )

        # If we hit the step limit and still have an open trade that did NOT
        # close via TP/SL above, force-close it mark-to-market at current tick.
        if hit_step_limit and not self.terminated:
            if self.position_side != 0 and self.position_entry_idx is not None and not closed_trade:
                exit_idx = self.t
                bid_now = float(self.tick_bid[exit_idx])
                ask_now = float(self.tick_ask[exit_idx])
                is_long = bool(self.position_is_long)

                self.position_duration_ticks += 1
                trade_duration_ticks = int(self.position_duration_ticks)

                base_pips = self._pnl_from_prices(
                    is_long,
                    self.position_entry_ask,
                    self.position_entry_bid,
                    bid_now,
                    ask_now,
                )
                pnl_pips = float(base_pips)
                pip_value = self._pip_value_usd()
                gross_profit_usd = float(pnl_pips * pip_value)

                # Swaps
                swap_usd = self._compute_swap_usd(is_long, self.position_entry_idx, exit_idx)

                # Commission & other costs
                commission_usd = self._commission_usd(round_turn=True)
                other_cost_usd = float(self.other_fixed_cost_per_trade_usd)

                net_profit_usd = gross_profit_usd + swap_usd - commission_usd - other_cost_usd

                # Track costs
                self.cumulative_commission += commission_usd
                self.cumulative_swap += swap_usd
                self.cumulative_other_costs += other_cost_usd

                # Update balance/equity & drawdown
                next_balance = self.balance + net_profit_usd
                if next_balance <= 0.0:
                    self.balance = 0.0
                    self.equity = 0.0
                    self.terminated = True
                else:
                    self.balance = float(next_balance)
                    self.equity = self._equity_mtm()

                if self.equity > self.equity_peak:
                    self.equity_peak = float(self.equity)
                if self.equity_peak > 0.0:
                    dd_now = max(0.0, (self.equity_peak - self.equity) / self.equity_peak)
                else:
                    dd_now = 0.0

                new_dd_increment = max(0.0, dd_now - self.max_drawdown)
                if dd_now > self.max_drawdown:
                    self.max_drawdown = float(dd_now)

                # 1R for logging
                one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
                profit_R = float(net_profit_usd / one_R_usd)

                closed_trade = True

                # Clear position
                self.position_side = 0
                self.position_entry_idx = None
                self.position_entry_ask = None
                self.position_entry_bid = None
                self.position_tp = None
                self.position_sl = None
                self.position_is_long = None
                self.position_duration_ticks = 0

            # Mark episode as truncated by step limit
            self.truncated = True

        # Final reward for this step
        if closed_trade:
            reward = self._calculate_reward(
                profit_usd=net_profit_usd,
                new_dd_increment=new_dd_increment,
            )
        else:
            reward = 0.0

        # Move to next tick if we are still alive
        if not self.terminated and not self.truncated:
            self.t += 1
            self.current_tick_row = self.t

        terminated_flag = bool(self.terminated)

        info = {

            "action": act,
            "final_idx": int(self.t),
            "balance": float(self.balance),
            "equity": float(self._equity_mtm()),
            "equity_peak": float(self.equity_peak),
            "dd_now": float(dd_now),
            "max_drawdown": float(self.max_drawdown),
            "new_dd_increment": float(new_dd_increment),
            "is_profitable": net_profit_usd >= 0.0 if closed_trade else False,
            "last_reward": float(reward),

            "profit_gross_usd": float(gross_profit_usd),
            "profit_net_usd": float(net_profit_usd),
            "profit": float(net_profit_usd),
            "profit_R": float(profit_R),
            "pnl_pips": float(pnl_pips),
            "swap_usd": float(swap_usd),
            "commission_usd": float(commission_usd),
            "other_cost_usd": float(other_cost_usd),
            "cumulative_commission": float(self.cumulative_commission),
            "cumulative_swap": float(self.cumulative_swap),
            "cumulative_other_costs": float(self.cumulative_other_costs),

            "trade_duration_ticks": int(trade_duration_ticks),
            "eof": bool(self.t >= self.n_ticks - 1),
            "closed_trade": closed_trade,
        }

        obs = self._get_observation(min(self.t, self.n_ticks - 1))
        return obs, float(reward), terminated_flag, bool(self.truncated), info

    # -------------------------------------------------------------------------
    # EOF handling (close any open trade MTM)
    # -------------------------------------------------------------------------
    def _handle_eof(self):
        closed_trade = False
        net_profit_usd = 0.0
        trade_duration_ticks = 0
        pnl_pips = 0.0
        gross_profit_usd = 0.0
        commission_usd = 0.0
        swap_usd = 0.0
        other_cost_usd = 0.0
        profit_R = 0.0
        dd_now = float(self.max_drawdown)
        new_dd_increment = 0.0

        if self.position_side != 0 and self.position_entry_idx is not None:
            exit_idx = self.n_ticks - 1
            bid = float(self.tick_bid[exit_idx])
            ask = float(self.tick_ask[exit_idx])

            is_long = bool(self.position_is_long)
            self.position_duration_ticks += 1
            trade_duration_ticks = int(self.position_duration_ticks)

            base_pips = self._pnl_from_prices(
                is_long,
                self.position_entry_ask,
                self.position_entry_bid,
                bid,
                ask,
            )

            pnl_pips = float(base_pips)
            pip_value = self._pip_value_usd()
            gross_profit_usd = float(pnl_pips * pip_value)

            # Swaps
            swap_usd = self._compute_swap_usd(is_long, self.position_entry_idx, exit_idx)

            # Commission & other costs
            commission_usd = self._commission_usd(round_turn=True)
            other_cost_usd = float(self.other_fixed_cost_per_trade_usd)

            net_profit_usd = gross_profit_usd + swap_usd - commission_usd - other_cost_usd

            # Track costs
            self.cumulative_commission += commission_usd
            self.cumulative_swap += swap_usd
            self.cumulative_other_costs += other_cost_usd

            # Update equity & drawdown
            next_balance = self.balance + net_profit_usd
            if next_balance <= 0.0:
                self.balance = 0.0
                self.equity = 0.0
                self.terminated = True
                self.truncated = True
            else:
                self.balance = float(next_balance)
                self.equity = self._equity_mtm()

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

            one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
            profit_R = float(net_profit_usd / one_R_usd)

            closed_trade = True

            # Clear position
            self.position_side = 0
            self.position_entry_idx = None
            self.position_entry_ask = None
            self.position_entry_bid = None
            self.position_tp = None
            self.position_sl = None
            self.position_is_long = None
            self.position_duration_ticks = 0

        reward = self._calculate_reward(
            profit_usd=net_profit_usd,
            new_dd_increment=new_dd_increment,
        )

        self.terminated = True

        info = {
            "action": "HOLD",
            "final_idx": int(self.n_ticks - 1),
            "balance": float(self.balance),
            "equity": float(self.equity),
            "equity_peak": float(self.equity_peak),
            "dd_now": float(dd_now),
            "max_drawdown": float(self.max_drawdown),
            "new_dd_increment": float(new_dd_increment),
            "is_profitable": net_profit_usd >= 0.0,
            "last_reward": float(reward),
            "profit_gross_usd": float(gross_profit_usd),
            "profit_net_usd": float(net_profit_usd),
            "profit": float(net_profit_usd),
            "profit_R": float(profit_R),
            "pnl_pips": float(pnl_pips),
            "swap_usd": float(swap_usd),
            "commission_usd": float(commission_usd),
            "other_cost_usd": float(other_cost_usd),
            "cumulative_commission": float(self.cumulative_commission),
            "cumulative_swap": float(self.cumulative_swap),
            "cumulative_other_costs": float(self.cumulative_other_costs),
            "trade_duration_ticks": int(trade_duration_ticks),
            "eof": True,
            "closed_trade": closed_trade,
            "reason": "eof",
        }
        return reward, info

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _update_price_normalizers(self, tick_idx: int):
        """
        Update running mean/std for ask, bid, spread used in price normalization.

        For simplicity we compute statistics over all ticks up to `tick_idx`
        (training region). This is called on each new bar, so it's not *every*
        tick, and it's fine performance-wise for offline training.
        """
        if not self.normalize_prices:
            return

        # Guard
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
        """
        Returns slippage in *pips* (>=0 for our worse-price convention).
        base_pips is the configured slippage_pips_*.
        """
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
        """Value of 1 pip (pip_decimal) in USD for 1.0 lot."""
        return (100000.0 * self.pip_decimal) / max(1e-12, self.exchange_rate)

    def _pip_value_usd(self) -> float:
        """Value of 1 pip in USD for the current lot size."""
        return self._pip_value_usd_per_lot() * self.lot

    def _commission_usd(self, round_turn: bool = False) -> float:
        """Commission in USD for this trade (fixed lot size)."""
        if not self.enable_commission or self.commission_per_lot_per_side_usd <= 0.0:
            return 0.0
        sides = 2 if round_turn else 1
        return float(self.lot * self.commission_per_lot_per_side_usd * sides)

    def _pnl_from_prices(
        self,
        is_long: bool,
        entry_ask: float,
        entry_bid: float,
        close_bid: float,
        close_ask: float,
    ) -> float:
        """
        Compute PnL in pips between entry and close prices:

          long:  buy at ask,   close at bid  -> profit if close_bid > entry_ask
          short: sell at bid,  close at ask  -> profit if close_ask < entry_bid
        """
        if is_long:
            # long: profit = close - entry
            return (close_bid - entry_ask) / self.pip_decimal
        else:
            # short: profit = entry - close
            return (entry_bid - close_ask) / self.pip_decimal


    def _compute_swap_usd(self, is_long: bool, entry_idx: int, exit_idx: int) -> float:
        """
        Approximate swap based on bar_times and tick_to_bar.
        swap_*_pips_per_day are per 1 lot; scaled by lot size via _pip_value_usd().
        """
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
                # assume seconds
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
        """
        Mark-to-market equity including floating PnL of the open position.
        Commissions and swaps are only applied on open/close, not MTM.
        """
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

    def _get_observation(self, tick_idx: int):
        """
        Observation vector:
            [window_len * normalized_bar_features] + [ask_norm, bid_norm, spread_norm]
        """
        bar_idx = int(self.tick_to_bar[tick_idx])
        L = self.window_len
        start = bar_idx - L + 1

        if start < 0:
            pad = np.zeros((-start, self.n_feats), dtype=np.float32)
            win = np.asarray(self.bar_features[0 : bar_idx + 1], dtype=np.float32)
            feat_win = np.vstack([pad, win])
        else:
            feat_win = np.asarray(self.bar_features[start : bar_idx + 1], dtype=np.float32)

        # Normalize bar features if scaler is available
        if self.normalize_bars and hasattr(self, "bars_mean") and self.bars_mean is not None:
            feat_win = (feat_win - self.bars_mean) / (self.bars_std + 1e-8)

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        # Normalized price features
        a = self._price_norm(ask, self.ask_mean, self.ask_std)
        b = self._price_norm(bid, self.bid_mean, self.bid_std)
        s = self._price_norm(spread, self.spread_mean, self.spread_std)
        price_feats = np.array([a, b, s], dtype=np.float32)

        # --- NEW: position-state features ---
        if self.position_side == 0 or self.position_entry_idx is None:
            # flat
            pos_dir = 0.0          # 0 = flat
            pos_pnl_R = 0.0        # unrealized PnL in R units
            pos_age = 0.0          # normalized trade age
        else:
            is_long = bool(self.position_is_long)
            # mark-to-market PnL in pips using same logic as equity_mtm
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
            # age scaled so typical trade lives in [0, 1] range
            pos_age = float(self.position_duration_ticks) / 1000.0

        pos_feats = np.array([pos_dir, pos_pnl_R, pos_age], dtype=np.float32)

        # Final observation = [bars window] + [price_feats] + [pos_feats]
        obs = np.hstack(
            [feat_win.ravel(), price_feats, pos_feats]
        ).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _calculate_reward(self, profit_usd: float, new_dd_increment: float) -> float:
        """
        Reward function for a single trade close (or forced MTM close).

        - Convert net profit in USD to R units, where 1R = SL distance in pips.
            This makes the reward scale invariant to lot size and symbol.

        - Slightly penalize losses more than equivalent gains:
                +1R  -> +1.0 reward
                -1R  -> -1.2 reward   (encourages avoiding bad trades)

        - Optionally penalize *increases* in max drawdown:
                reward -= dd_penalty_lambda * new_dd_increment

            where new_dd_increment is in [0, 1] (fractional equity drawdown).
        """
        # If no movement, zero reward
        if profit_usd == 0.0:
            base_reward = 0.0
        else:
            # 1R in USD: SL (in pips) * pip_value_per_lot * lot_size
            one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
            profit_R = float(profit_usd / one_R_usd)

            # Asymmetric shaping: make losses hurt slightly more
            if profit_R > 0.0:
                base_reward = profit_R
            else:
                base_reward = 1.2 * profit_R  # amplify loss a bit

        # Drawdown penalty (only when DD increases)
        dd_penalty = self.dd_penalty_lambda * float(new_dd_increment)

        reward = base_reward - dd_penalty
        return float(reward)

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
