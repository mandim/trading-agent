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
    Merged DQN-ready trading environment with broker costs.

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
      - Broker cost model:
            * Spread (optional, via include_spread_cost)
            * Commission (per-lot per-side, account-type aware)
            * Swaps (pips/day for long & short)
            * Slippage (pips on open/close)
            * Other fixed per-trade costs
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
        include_spread_cost: bool = True,
        exchange_rate: float = 1.0,   # 1.0 for USD-quoted symbols like EURUSD

        # Broker cost config
        account_type: str = "standard",        # "raw" or "standard"
        commission_per_lot_per_side_usd: float | None = None,
        enable_commission: bool = True,
        enable_swaps=True,
        swap_long_pips_per_day=-1.3,
        swap_short_pips_per_day=0.42,
        slippage_pips_open: float = 1.0,
        slippage_pips_close: float = 1.0,
        slippage_mode: str = "normal",   # "fixed", "uniform", "normal"
        other_fixed_cost_per_trade_usd: float = 0.0,

        # Risk & penalties
        dd_penalty_lambda: float = 1.0,
        max_dd_stop: float = 0.30,

        # Trade life control (NEW)
        max_trade_ticks: int | None = None,  # e.g. 5000; None = no time-based close

        # Reward configuration
        reward_mode: str = "risk",    # "risk" or "pnl"
        risk_per_trade_usd: float = 1000.0,
        reward_dense: bool = False,   # NEW: if True, use Δequity per step
        
        # Position sizing (NEW)
        risk_percent: float = 0.0,        # e.g. 1.0 = 1% of equity
        use_percent_risk: bool = False,   # if True, ignore risk_per_trade_usd
        min_lot: float = 0.01,
        max_lot: float = 100.0,

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

        # Paths
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

        # Broker cost config
        self.account_type = account_type.lower()
        if commission_per_lot_per_side_usd is None:
            # FP Markets-like defaults:
            # Raw: ~$3 per lot per side; Standard: 0 (in spread)
            self.commission_per_lot_per_side_usd = 3.0 if self.account_type == "raw" else 0.0
        else:
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

        # Trade life control (NEW)
        self.max_trade_ticks = max_trade_ticks

        # Reward
        self.reward_mode = str(reward_mode)
        self.risk_per_trade_usd = float(risk_per_trade_usd)
        self.reward_dense = bool(reward_dense)   # NEW
        
        # Position sizing
        self.risk_percent = float(risk_percent)
        self.use_percent_risk = bool(use_percent_risk)
        self.min_lot = float(min_lot)
        self.max_lot = float(max_lot)

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
        self.prev_equity: float | None = None   # NEW

        # Cost trackers
        self.cumulative_commission = 0.0
        self.cumulative_swap = 0.0
        self.cumulative_other_costs = 0.0
        
        # MT4-style persistent position state
        # 0 = flat, +1 = long, -1 = short
        self.position_side = 0
        self.position_entry_idx = None
        self.position_entry_ask = None
        self.position_entry_bid = None
        self.position_tp = None
        self.position_sl = None
        self.position_spread_once_pips = 0.0
        self.position_is_long = None
        self.position_duration_ticks = 0

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

        # Reset state
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
        
        # Reset position state
        self.position_side = 0
        self.position_entry_idx = None
        self.position_entry_ask = None
        self.position_entry_bid = None
        self.position_tp = None
        self.position_sl = None
        self.position_spread_once_pips = 0.0
        self.position_is_long = None
        self.position_duration_ticks = 0

        # Choose starting tick (deterministic MT4-style evaluation)
        min_bar = self.window_len - 1
        base = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        if base >= self.n_ticks:
            raise ValueError("Warmup exceeds available ticks; check window_len or data.")

        if self.eval_mode:
            # MT4-style behavior: always start at the first valid tick of the EVAL region
            # No randomness, no resampling. A single continuous backtest.
            split_tick = int(self.n_ticks * self.train_fraction)
            self.t = max(base, split_tick)
        else:
            # Training mode keeps randomization
            split_tick = int(self.n_ticks * self.train_fraction)
            start_low = base
            start_high = max(start_low + 1, max(base + 1, split_tick))
            if start_low >= start_high:
                self.t = base
            else:
                self.t = int(self._rng.integers(low=start_low, high=start_high, endpoint=False))

        self.current_tick_row = self.t

        # Initialize price normalizers using history up to starting tick
        self._update_price_normalizers(self.t)
        
        # Dense reward baseline: equity at start state
        self.prev_equity = self._equity_mtm()    # NEW

        obs = self._get_observation(self.t)
        return obs, {}

    def step(self, action: int):
        # If episode is already over, no-op
        if self.terminated or self.truncated:
            return self._get_observation(self.t), 0.0, self.terminated, self.truncated, {
                "note": "episode already ended"
            }

        # End-of-data guard: if we are at or beyond last tick, force-close any open trade and end.
        if self.t >= self.n_ticks - 1:
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

            # Force-close open position at EOF at worst case (treat as SL-ish)
            if self.position_side != 0 and self.position_entry_idx is not None:
                exit_idx = self.n_ticks - 1
                bid = float(self.tick_bid[exit_idx])
                ask = float(self.tick_ask[exit_idx])

                is_long = bool(self.position_is_long)
                self.position_duration_ticks += 1
                trade_duration_ticks = int(self.position_duration_ticks)

                # Closing price & PnL in pips
                if is_long:
                    close_px = bid
                    base_pips = -self.sl_pips  # treat EOF as SL
                    direction_sign = 1.0
                else:
                    close_px = ask
                    base_pips = -self.sl_pips
                    direction_sign = -1.0

                pnl_pips = float(base_pips)
                if self.include_spread_cost and self.position_spread_once_pips > 0.0:
                    pnl_pips -= float(self.position_spread_once_pips)

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

                # 1R reference
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
                self.position_spread_once_pips = 0.0
                self.position_is_long = None
                self.position_duration_ticks = 0

                reward = self._calculate_reward(
                    profit_usd=net_profit_usd,
                    new_dd_increment=new_dd_increment,
                )
            else:
                reward = 0.0
                
            # NEW: dense reward mode -> use Δequity instead of per-trade PnL
            if self.reward_dense:
                eq_now = self._equity_mtm()
                if self.prev_equity is None:
                    delta_eq = 0.0
                else:
                    delta_eq = eq_now - self.prev_equity
                self.prev_equity = eq_now
                reward = self._calculate_reward(
                    profit_usd=delta_eq,
                    new_dd_increment=new_dd_increment,
                )

            self.terminated = True
            obs = self._get_observation(self.n_ticks - 1)
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
                "reward_mode": self.reward_mode,
                "reason": "eof",
            }
            return obs, float(reward), True, bool(self.truncated), info

        # Determine if this tick is the first tick of its bar
        cur_bar = int(self.tick_to_bar[self.t])
        prev_bar = int(self.tick_to_bar[self.t - 1]) if self.t > 0 else -1
        is_new_bar = (cur_bar != prev_bar)

        # On every new bar, update price normalizers (dynamic normalization)
        # so env matches MT4 server behavior.
        if is_new_bar:
            self._update_price_normalizers(self.t)

        # In eval_mode, only allow decisions on bar open.
        # For all intra-bar ticks, force HOLD so behavior matches MT4 D1 EA execution.
        if self.eval_mode and not is_new_bar:
            action = 0  # HOLD
        
        # Map action id -> label
        if action == 0:
            act = "HOLD"
        elif action == 1:
            act = "BUY"
        elif action == 2:
            act = "SELL"
        else:
            act = "HOLD"

        # Current tick prices
        bid = float(self.tick_bid[self.t])
        ask = float(self.tick_ask[self.t])

        closed_trade = False
        net_profit_usd = 0.0
        gross_profit_usd = 0.0
        pnl_pips = 0.0
        commission_usd = 0.0
        swap_usd = 0.0
        other_cost_usd = 0.0
        profit_R = 0.0
        trade_duration_ticks = 0
        dd_now = float(self.max_drawdown)
        new_dd_increment = 0.0

        # ------------------------------------------------------------------
        # 1) Check TP/SL for existing position at this tick
        # ------------------------------------------------------------------
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
            
            # NEW: time-based exit if trade lives too long
            if exit_reason is None and self.max_trade_ticks is not None:
                if self.position_duration_ticks >= self.max_trade_ticks:
                    exit_reason = "time"

            if exit_reason is not None:
                exit_idx = self.t

                # Closing price before slippage (for logging)
                if is_long:
                    if exit_reason == "tp":
                        close_px = self.position_tp
                    elif exit_reason == "sl":
                        close_px = self.position_sl
                    else:  # "time" -> close at current BID
                        close_px = bid
                else:
                    if exit_reason == "tp":
                        close_px = self.position_tp
                    elif exit_reason == "sl":
                        close_px = self.position_sl
                    else:  # "time" -> close at current ASK
                        close_px = ask

                # Apply closing slippage
                if self.slippage_pips_close > 0.0:
                    slip_close_pips = self._sample_slippage(self.slippage_pips_close)
                    slip_close = slip_close_pips * self.pip_decimal
                    if is_long:
                        close_px -= slip_close   # sell lower on close
                    else:
                        close_px += slip_close   # buy higher on close

                # Base PnL in pips:
                # - "tp"/"sl": fixed ±TP/SL (for clean R accounting)
                # - "time": true mark-to-market pips from entry to close_px
                if exit_reason == "tp":
                    base_pips = self.tp_pips
                elif exit_reason == "sl":
                    base_pips = -self.sl_pips
                else:  # "time"
                    if is_long:
                        base_pips = (close_px - self.position_entry_ask) / self.pip_decimal
                    else:
                        base_pips = (self.position_entry_bid - close_px) / self.pip_decimal

                pnl_pips = float(base_pips)
                if self.include_spread_cost and self.position_spread_once_pips > 0.0:
                    pnl_pips -= float(self.position_spread_once_pips)

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

                # 1R reference
                one_R_usd = max(1e-8, self._pip_value_usd() * self.sl_pips)
                profit_R = float(net_profit_usd / one_R_usd)

                closed_trade = True

                # Clear position state after close
                self.position_side = 0
                self.position_entry_idx = None
                self.position_entry_ask = None
                self.position_entry_bid = None
                self.position_tp = None
                self.position_sl = None
                self.position_spread_once_pips = 0.0
                self.position_is_long = None
                self.position_duration_ticks = 0

        # ------------------------------------------------------------------
        # 2) Process action (open new trade only if flat)
        # ------------------------------------------------------------------
        if self.position_side == 0 and not self.terminated and not self.truncated:
            if act == "BUY":
                # --- NEW: compute lot size from current equity / settings ---
                self.lot = self._compute_lot_for_new_trade()
                
                # Open long at ask with opening slippage
                entry_idx = self.t
                entry_ask = ask
                entry_bid = bid

                if self.slippage_pips_open > 0.0:
                    slip_open_pips = self._sample_slippage(self.slippage_pips_open)
                    slip_open = slip_open_pips * self.pip_decimal
                    entry_ask += slip_open  # pay more to buy

                raw_spread_pips = max(0.0, (entry_ask - entry_bid) / self.pip_decimal)
                if self.account_type == "standard":
                    spread_once_pips = 1.8
                else:
                    spread_once_pips = raw_spread_pips

                tp_value = entry_ask + self.tp_pips * self.pip_decimal
                sl_value = entry_bid - self.sl_pips * self.pip_decimal

                self.position_side = 1
                self.position_entry_idx = entry_idx
                self.position_entry_ask = entry_ask
                self.position_entry_bid = entry_bid
                self.position_tp = tp_value
                self.position_sl = sl_value
                self.position_spread_once_pips = float(spread_once_pips)
                self.position_is_long = True
                self.position_duration_ticks = 0

            elif act == "SELL":
                # --- NEW: compute lot size from current equity / settings ---
                self.lot = self._compute_lot_for_new_trade()
                
                # Open short at bid with opening slippage
                entry_idx = self.t
                entry_ask = ask
                entry_bid = bid

                if self.slippage_pips_open > 0.0:
                    slip_open_pips = self._sample_slippage(self.slippage_pips_open)
                    slip_open = slip_open_pips * self.pip_decimal
                    entry_bid -= slip_open  # receive less to sell

                raw_spread_pips = max(0.0, (entry_ask - entry_bid) / self.pip_decimal)
                if self.account_type == "standard":
                    spread_once_pips = 1.8
                else:
                    spread_once_pips = raw_spread_pips

                tp_value = entry_bid - self.tp_pips * self.pip_decimal
                sl_value = entry_ask + self.sl_pips * self.pip_decimal

                self.position_side = -1
                self.position_entry_idx = entry_idx
                self.position_entry_ask = entry_ask
                self.position_entry_bid = entry_bid
                self.position_tp = tp_value
                self.position_sl = sl_value
                self.position_spread_once_pips = float(spread_once_pips)
                self.position_is_long = False
                self.position_duration_ticks = 0
            # HOLD when flat: do nothing, just move to next tick

        # ------------------------------------------------------------------
        # 3) Reward (per closed trade) & step bookkeeping
        # ------------------------------------------------------------------
        if closed_trade:
            reward = self._calculate_reward(
                profit_usd=net_profit_usd,
                new_dd_increment=new_dd_increment,
            )
        else:
            reward = 0.0

        self.steps += 1
        if self.max_steps_per_episode is not None and self.steps >= self.max_steps_per_episode:
            self.truncated = True

        # Move to next tick (unless we truncated/terminated by DD/etc)
        if not self.terminated and not self.truncated:
            self.t += 1
            self.current_tick_row = self.t
            
        # NEW: dense reward mode -> reward = Δequity from previous state to new state
        if self.reward_dense:
            eq_now = self._equity_mtm()
            if self.prev_equity is None:
                delta_eq = 0.0
            else:
                delta_eq = eq_now - self.prev_equity
            self.prev_equity = eq_now

            reward = self._calculate_reward(
                profit_usd=delta_eq,
                new_dd_increment=new_dd_increment,
            )

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
            "reward_mode": self.reward_mode,
        }

        obs = self._get_observation(min(self.t, self.n_ticks - 1))
        return obs, float(reward), terminated_flag, bool(self.truncated), info

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
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
            # Uniform [0, base_pips]
            return float(self._rng.uniform(0.0, base_pips))

        if mode == "normal":
            # Half-normal around 0 with sigma = base_pips
            val = abs(self._rng.normal(loc=0.0, scale=base_pips))
            return float(val)

        # Fallback
        return base_pips

    def _pip_value_usd_per_lot(self) -> float:
        """
        Value of 1 pip (pip_decimal) in USD for 1.0 lot.
        """
        return (100000.0 * self.pip_decimal) / max(1e-12, self.exchange_rate)

    def _compute_lot_for_new_trade(self) -> float:
        """
        Compute lot size for the next trade.

        If use_percent_risk=True and risk_percent>0:
            lot is chosen so that:
                risk_usd ≈ equity * (risk_percent / 100) ≈ sl_pips * pip_value_per_lot * lot
        Otherwise:
            fall back to fixed risk_per_trade_usd (same as training setup).
        """
        # Choose risk in USD
        if self.use_percent_risk and self.risk_percent > 0.0:
            risk_usd = self.equity * (self.risk_percent / 100.0)
        else:
            risk_usd = self.risk_per_trade_usd

        pip_value_per_lot = self._pip_value_usd_per_lot()
        if pip_value_per_lot <= 0.0 or self.sl_pips <= 0.0 or risk_usd <= 0.0:
            # fall back to existing lot
            lot = self.lot
        else:
            lot = risk_usd / (self.sl_pips * pip_value_per_lot)

        # Clamp to sensible bounds
        lot = max(self.min_lot, min(self.max_lot, lot))
        return float(lot)

    def _pip_value_usd(self) -> float:
        """
        Value of 1 pip (pip_decimal) in USD for the *current* lot size.
        """
        return self._pip_value_usd_per_lot() * self.lot

    def _commission_usd(self, round_turn: bool = False) -> float:
        """
        Commission in USD for this trade (fixed lot size).
        If round_turn=True, charges both open and close.
        """
        if not self.enable_commission or self.commission_per_lot_per_side_usd <= 0.0:
            return 0.0
        sides = 2 if round_turn else 1
        return float(self.lot * self.commission_per_lot_per_side_usd * sides)

    def _update_price_normalizers(self, tick_idx: int):
        """
        Update ask/bid/spread mean/std based on a rolling tick window up to tick_idx.
        Mirrors the per-bar normalization your MT4 ZMQ server does.
        """
        if (not self.normalize_prices) or tick_idx <= 0:
            self.ask_mean = self.ask_std = None
            self.bid_mean = self.bid_std = None
            self.spread_mean = self.spread_std = None
            return

        # same logic as in reset, but reusable
        lookback = max(0, tick_idx - 20_000)
        if tick_idx <= lookback:
            self.ask_mean = self.ask_std = None
            self.bid_mean = self.bid_std = None
            self.spread_mean = self.spread_std = None
            return

        ask_slice = np.asarray(self.tick_ask[lookback:tick_idx + 1], dtype=np.float32)
        bid_slice = np.asarray(self.tick_bid[lookback:tick_idx + 1], dtype=np.float32)
        spread_slice = ask_slice - bid_slice

        self.ask_mean = float(ask_slice.mean())
        self.ask_std = float(ask_slice.std() + 1e-8)
        self.bid_mean = float(bid_slice.mean())
        self.bid_std = float(bid_slice.std() + 1e-8)
        self.spread_mean = float(spread_slice.mean())
        self.spread_std = float(spread_slice.std() + 1e-8)

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

        # Support datetime64 or numeric seconds
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

        # rate is pips/day per 1 lot; _pip_value_usd() already includes lot size
        return float(rate * self._pip_value_usd() * days_held)

    def _equity_mtm(self) -> float:
        """
        Mark-to-market equity including floating PnL of the open position (if any).
        Commissions are applied on open/close only; swaps are only applied on close.
        """
        eq = float(self.balance)

        # If flat, equity == balance
        if self.position_side == 0 or self.position_entry_idx is None:
            return eq

        # Use current tick for pricing (clamped to data range)
        idx = min(max(self.t, 0), self.n_ticks - 1)
        bid = float(self.tick_bid[idx])
        ask = float(self.tick_ask[idx])
        is_long = bool(self.position_is_long)

        if is_long:
            close_px = bid
            base_pips = (close_px - self.position_entry_ask) / self.pip_decimal
        else:
            close_px = ask
            base_pips = (self.position_entry_bid - close_px) / self.pip_decimal

        pnl_pips = float(base_pips)
        if self.include_spread_cost and self.position_spread_once_pips > 0.0:
            pnl_pips -= float(self.position_spread_once_pips)

        pip_value = self._pip_value_usd()
        float_profit_usd = float(pnl_pips * pip_value)

        return float(eq + float_profit_usd)


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
        Unified reward using NET profit_usd:
          - "risk": reward = (profit_usd / denom) - λ * ΔDD
                    denom = risk_per_trade_usd (if >0) else 1R
          - "pnl" : reward = (profit_usd / denom) - λ * ΔDD
                    denom = risk_per_trade_usd (if >0) else 1.0
          - else : fallback to 1R-based normalization.
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
