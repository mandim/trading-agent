import pandas as pd
from server import ZMQRepServer
import talib as ta
import numpy as np

class TradingEnv:

    def __init__(self, pip_decimal: float, candles_file: str, tick_file: str, bind_address="tcp://*:5555"):
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.current_tick_row = 0
        self.position_open = False
        self.server = ZMQRepServer(bind_address, self._handle_request)
        self.pip_decimal = pip_decimal
        self.tp_pips = 50.0
        self.sl_pips = 50.0
        self.lot = 1.0
        self.balance = 100000.0
        self.truncated = False
        self.done = False
        self.risk_per_trade_usd = 1000.0   # denominator for PnL normalization (e.g., ~1% of 100k)
        self.dd_penalty_lambda = 1.0       # how strongly to punish new drawdown (tune 0.3–2.0)
        self.max_dd_stop = 0.30            # hard stop at 30% drawdown (optional)
        
        # RUNTIME (also set these in reset)
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0

        # ---- BAR/TICK FUSION CONFIG ----
        self.window_len = 32  # number of bars to include in the observation window

        # Load bars + indicators and build tick->bar alignment
        self._load_candles_and_indicators()
        self._compute_warmup()
        self._prepare_tick_bar_alignment()

    def start_server(self):
        print("Starting Env Server...")
        self.server.start()
    
    def stop_server(self):
        print("Stoping Env Server...")
        self.server.stop()

    def _handle_request(self, request):
        """
        Process incoming request and return a reply.
        Request is expected to be a dict; adapt as needed.
        """
        # Simple example: support {"cmd": "ping"} and {"cmd": "sum", "values": [..]}
        cmd = request.get("cmd") if isinstance(request, dict) else None

        if cmd == "BUY":
            self.step("BUY")
            return {"reply": "BUY position closed and return current state to the Agent"}
        elif cmd == "SELL":
            self.step("SELL")
            return {"reply": "SELL position closed and return current state to the Agent"}
        elif cmd == "HOLD":
            return {"reply": "Hold position and continue"}
        else:
            # default: echo
            return {"reply": "unknown_command", "received": request}

    # --- helper generator that yields rows with a global index ---
    def _iter_ticks_from(self, start_row=0, chunksize=1000):
        """
        Iterate rows from `start_row` (absolute row index in file).
        Yields (global_idx, row) pairs where global_idx is the absolute row index.
        """
        global_start = 0
        for chunk in pd.read_csv(self.tick_file, chunksize=chunksize):
            chunk_len = len(chunk)
            chunk_end = global_start + chunk_len
            # if entire chunk is before start_row, skip it quickly
            if chunk_end <= start_row:
                global_start = chunk_end
                continue
            # iterate rows in chunk and compute absolute index
            for i, (_, row) in enumerate(chunk.iterrows()):
                global_idx = global_start + i
                if global_idx < start_row:
                    continue
                yield global_idx, row
            global_start = chunk_end

    def _get_tick_row(self):
        """
        Return (idx, row) for the current_tick_row or (None, None) if EOF.
        """
        for idx, row in self._iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
            # the first yielded row will be the one at current_tick_row (if exists)
            return idx, row
        return None, None
    
    def reset(self):
        self.balance = 100000.0
        if len(self.abs_tick_indices) == 0:
            raise ValueError("No valid ticks after warmup. Check data/time parsing and warmup settings.")
        self.current_tick_row = int(self.abs_tick_indices[0])  # start at first valid aligned tick
        self.position_open = False
        self.truncated = False
        self.done = False
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0

        # Build initial observation (first tick or first bar/indicators)
        obs = self._get_observation(self.current_tick_row)
        return obs, {}
    
    def step(self, action: str):
        """
        Event-driven per-trade step.
        Actions: "BUY", "SELL", "HOLD"
        Returns: obs, reward, terminated, truncated, info  (Gymnasium-compatible)
        """
        if self.done:
            # Safe terminal repeat
            obs = self._get_observation(self.current_tick_row)
            return obs, 0.0, True, self.truncated, {"note": "episode already done"}

        # Defaults
        tp_value = sl_value = None
        is_profitable = None

        # HOLD: advance one tick, no trade opened
        if action == "HOLD":
            # Move pointer by 1 if possible
            idx, row = self._get_tick_row()
            if idx is None:
                self.done = True
                obs = self._get_observation(self.current_tick_row)
                return obs, 0.0, True, self.truncated, {"action": "HOLD", "reason": "eof"}
            # advance pointer to next row and return 0 reward
            self.current_tick_row = idx + 1
            obs = self._get_observation(self.current_tick_row)
            return obs, 0.0, False, self.truncated, {"action": "HOLD"}

        # Open position
        self.position_open = True

        # ---------- SELL (short) ----------
        if action == "SELL":
            idx, tick_row = self._get_tick_row()
            if idx is None:
                self.position_open = False
                self.done = True
                obs = self._get_observation(self.current_tick_row)
                return obs, 0.0, True, self.truncated, {"action": "SELL", "reason": "eof_before_entry"}

            bid_value = float(tick_row["Bid price"])
            # advance pointer past entry tick
            self.current_tick_row = idx + 1

            # exit thresholds (exit at ask when covering a short, but we check levels on bid
            # to avoid lookahead since fills happen on ask side but barrier decision uses best available)
            tp_value = bid_value - (self.tp_pips * self.pip_decimal)
            sl_value = bid_value + (self.sl_pips * self.pip_decimal)

            for idx, row in self._iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                bid_price = float(row["Bid price"])
                # move pointer forward
                self.current_tick_row = idx + 1

                if bid_price <= tp_value:
                    self.position_open = False
                    is_profitable = True
                    break
                elif bid_price >= sl_value:  # exact-touch counts (fixed)
                    self.position_open = False
                    is_profitable = False
                    break

            if self.position_open:
                # EOF without TP/SL
                self.position_open = False
                self.done = True
                # fall through to reward calc (treat as flat PnL from threshold distance if you want;
                # here we just compute via your _calculate_reward using TP/SL choice below)
                # choose a conservative close: treat as SL
                is_profitable = False

        # ---------- BUY (long) ----------
        elif action == "BUY":
            idx, tick_row = self._get_tick_row()
            if idx is None:
                self.position_open = False
                self.done = True
                obs = self._get_observation(self.current_tick_row)
                return obs, 0.0, True, self.truncated, {"action": "BUY", "reason": "eof_before_entry"}

            ask_value = float(tick_row["Ask price"])
            # advance pointer past entry tick
            self.current_tick_row = idx + 1

            tp_value = ask_value + (self.tp_pips * self.pip_decimal)
            sl_value = ask_value - (self.sl_pips * self.pip_decimal)

            for idx, row in self._iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                ask_price = float(row["Ask price"])
                # move pointer forward
                self.current_tick_row = idx + 1

                if ask_price >= tp_value:
                    self.position_open = False
                    is_profitable = True
                    break
                elif ask_price <= sl_value:
                    self.position_open = False
                    is_profitable = False
                    break

            if self.position_open:
                # EOF without TP/SL
                self.position_open = False
                self.done = True
                # conservative close: treat as SL
                is_profitable = False

        else:
            # Unknown action: no-op advance 1 tick
            idx, row = self._get_tick_row()
            if idx is None:
                self.done = True
                obs = self._get_observation(self.current_tick_row)
                return obs, 0.0, True, self.truncated, {"action": action, "reason": "eof"}
            self.current_tick_row = idx + 1
            obs = self._get_observation(self.current_tick_row)
            return obs, 0.0, False, self.truncated, {"action": action, "note": "unknown_action_noop"}

        # ----- Compute reward (DD-aware) -----
        reward = self._calculate_reward(isProfitable=is_profitable)

        # ----- Build obs & info -----
        obs = self._get_observation(self.current_tick_row)
        info = {
            "action": action,
            "tp": tp_value,
            "sl": sl_value,
            "final_idx": self.current_tick_row,
            "balance": self.balance,
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "max_drawdown": self.max_drawdown,
            "is_profitable": is_profitable,
            "last_reward": reward,
        }

        return obs, float(reward), self.done, self.truncated, info

    def _get_observation(self, idx):
        """
        Build observation at absolute tick row 'idx':
        - live tick features: ask, bid, spread
        - window_len bars of indicators up to the most recent CLOSED bar
        """
        # Find nearest kept tick position <= idx (no lookahead)
        pos = np.searchsorted(self.abs_tick_indices, idx, side="right") - 1
        if pos < 0:
            pos = 0
        if pos >= len(self.abs_tick_indices):
            # Past the last aligned tick: return zeros
            ask = bid = spread = 0.0
            bar_idx = len(self.bars) - 1
        else:
            abs_tick_idx = int(self.abs_tick_indices[pos])

            # Read that tick’s prices via your generator (first row from abs_tick_idx)
            ask = bid = None
            for t_idx, row in self._iter_ticks_from(start_row=abs_tick_idx, chunksize=1000):
                ask = float(row["Ask price"])
                bid = float(row["Bid price"])
                break
            if ask is None or bid is None:
                ask = bid = 0.0
            spread = ask - bid

            bar_idx = int(self.bar_idx_for_tick[pos])

        # Slice bar window [bar_idx - L + 1, bar_idx]
        L = self.window_len
        start = max(0, bar_idx - L + 1)
        w = self.bars.iloc[start:bar_idx+1]

        def pad(series):
            a = series.values.astype(np.float32)
            if len(a) < L:
                a = np.hstack([np.zeros(L - len(a), dtype=np.float32), a])
            return a

        obs = np.hstack([
            pad(w["ret1"]),
            pad(w["sma50"]),
            pad(w["ema12"]),
            pad(w["ema26"]),
            pad(w["macd"]),
            pad(w["macd_sig"]),
            pad(w["rsi14"]),
            pad(w["atr14"]),
            np.array([ask, bid, spread, float(self.position_open)], dtype=np.float32),
        ])
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _calculate_reward(self, isProfitable: bool):
        # --- 1) Compute trade PnL ---
        old_balance = self.balance
        exchange_rate = 1.0  # adjust if needed
        pip_value = ((self.lot * 100000) * self.pip_decimal) / exchange_rate
        profit = 0.0
        if isProfitable is True:
            profit = pip_value * self.tp_pips
        elif isProfitable is False:
            profit = -pip_value * self.sl_pips

        # --- 2) Update balance & equity ---
        next_balance = old_balance + profit
        if next_balance <= 0:
            self.truncated = True
            self.done = True
            self.balance = 0.0
        else:
            self.balance = next_balance

        old_equity = self.equity
        self.equity = self.balance  # equity = balance here (no floating PnL)

        # --- 3) Update drawdown metrics ---
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity
        dd_now = 0.0 if self.equity_peak == 0 else (self.equity_peak - self.equity) / self.equity_peak
        dd_now = max(0.0, dd_now)
        new_dd_increment = max(0.0, dd_now - self.max_drawdown)
        if dd_now > self.max_drawdown:
            self.max_drawdown = dd_now

        # --- 4) Normalize profit and apply DD penalty ---
        denom = max(1e-8, self.risk_per_trade_usd)
        pnl_norm = profit / denom
        penalty = self.dd_penalty_lambda * new_dd_increment
        reward = pnl_norm - penalty

        # --- 5) Hard stop if max DD breached ---
        if self.max_drawdown >= self.max_dd_stop:
            self.done = True
            self.truncated = True

        return float(reward)

    def _ensure_datetime(self, ds, ts):
        """
        Bars have separate Date (YYYYMMDD or YYYY-MM-DD) and Timestamp (HH:MM:SS).
        Returns a pandas datetime series.
        """
        ds = pd.to_datetime(ds.astype(str), format="%Y%m%d", errors="coerce", utc=False)
        # If your Date is already like '2020-09-15', the above will be NaT; try auto-parse:
        if ds.isna().any():
            ds = pd.to_datetime(ds, errors="coerce", utc=False)
        # combine date + time
        ts = pd.to_datetime(ts.astype(str), format="%H:%M:%S", errors="coerce", utc=False).dt.time
        return pd.to_datetime(ds.dt.date.astype(str) + " " + pd.Series(ts).astype(str), utc=False)

    def _load_candles_and_indicators(self):
        """
        Load candle CSV with columns: Date, Timestamp, Open, High, Low, Close, Volume
        Compute TA-Lib indicators on Close/High/Low.
        Stores result in self.bars with a 'bar_time' column (the bar close time).
        """
        bars = pd.read_csv(self.candles_file)
        # normalize column names (case-sensitive exact names as per your sample)
        required = ["Date","Timestamp","Open","High","Low","Close"]
        for c in required:
            if c not in bars.columns:
                raise ValueError(f"Missing '{c}' in candles file.")

        bars["bar_time"] = self._ensure_datetime(bars["Date"], bars["Timestamp"])
        bars = bars.sort_values("bar_time").reset_index(drop=True)

        close = bars["Close"].astype(float).values
        high  = bars["High"].astype(float).values
        low   = bars["Low"].astype(float).values

        # --- TA-Lib indicators ---
        bars["sma50"]     = ta.SMA(close, timeperiod=50)
        bars["ema12"]     = ta.EMA(close, timeperiod=12)
        bars["ema26"]     = ta.EMA(close, timeperiod=26)
        macd, macd_sig, _ = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        bars["macd"]      = macd
        bars["macd_sig"]  = macd_sig
        bars["rsi14"]     = ta.RSI(close, timeperiod=14)
        bars["atr14"]     = ta.ATR(high, low, close, timeperiod=14)

        # simple bar return for context
        bars["ret1"]      = pd.Series(close).pct_change().fillna(0.0)

        self.bars = bars  # keep all columns

    def _compute_warmup(self):
        """
        Longest indicator lookback + window padding to avoid NaNs in obs.
        EMA needs extra stabilization.
        """
        raw_lookback   = max(50, 26, 14)  # SMA50, EMA26, RSI14
        ema_stabilize  = 3 * 26           # ~3× longest EMA period
        self.warmup_bars = max(raw_lookback, ema_stabilize) + (self.window_len - 1)

    def _parse_tick_time(self, df):
        """
        Tries to build a datetime column named 'tick_time' from ticks df.
        Supports:
        - Date + Timestamp columns (like bars)
        - or a single Timestamp containing full datetime
        """
        if "Date" in df.columns and "Timestamp" in df.columns:
            tick_time = self._ensure_datetime(df["Date"], df["Timestamp"])
        else:
            # try to parse a single full datetime column
            tcol = None
            for c in ["Timestamp","timestamp","Time","time","Datetime","datetime"]:
                if c in df.columns:
                    tcol = c
                    break
            if tcol is None:
                raise ValueError("No time column found in tick file.")
            tick_time = pd.to_datetime(df[tcol], errors="coerce", utc=False)
        if tick_time.isna().any():
            raise ValueError("Failed to parse some tick timestamps.")
        return tick_time

    def _prepare_tick_bar_alignment(self):
        """
        Build arrays:
        self.ticks_meta            - minimal tick dataframe with tick_time
        self.bar_idx_for_tick[i]  - bar index for the i-th kept tick
        self.abs_tick_indices[i]  - absolute row index in the original tick CSV for that kept tick
        """
        ticks = pd.read_csv(self.tick_file)
        ticks["tick_time"] = self._parse_tick_time(ticks)
        ticks = ticks.sort_values("tick_time").reset_index(drop=True)

        bar_times  = self.bars["bar_time"].values.astype("datetime64[ns]")
        tick_times = ticks["tick_time"].values.astype("datetime64[ns]")

        # For each tick, find rightmost bar with bar_time <= tick_time
        bar_idx_for_tick = np.searchsorted(bar_times, tick_times, side="right") - 1

        # Keep only ticks that occur after warmup bars
        valid = bar_idx_for_tick >= self.warmup_bars
        self.ticks_meta      = ticks.loc[valid, ["tick_time"]].reset_index(drop=True)
        self.bar_idx_for_tick = bar_idx_for_tick[valid]
        # map to absolute indices (row ids in the original sorted ticks df)
        self.abs_tick_indices = np.where(valid)[0].astype(int)
