import pandas as pd
from server import ZMQRepServer
import talib as ta
import math

class TradingEnv:

    def __init__(self, pip_decimal: float, candles_file: str, tick_file: str, bind_address="tcp://*:5555"):
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.current_tick_row = 0
        self.position_open = False
        self.server = ZMQRepServer(bind_address, self.handle_request)
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

    def start_server(self):
        print("Starting Env Server...")
        self.server.start()
    
    def stop_server(self):
        print("Stoping Env Server...")
        self.server.stop()

    def handle_request(self, request):
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
    def iter_ticks_from(self, start_row=0, chunksize=1000):
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

    def get_tick_row(self):
        """
        Return (idx, row) for the current_tick_row or (None, None) if EOF.
        """
        for idx, row in self.iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
            # the first yielded row will be the one at current_tick_row (if exists)
            return idx, row
        return None, None
    
    def reset(self):
        self.balance = 100000
        self.current_tick_row = 0
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
        This is the step function that opens a new position. The action is the position of BUY, SHORT or HOLD.
        The step returns a new state of OHLCV as a new observation along with a reward that is based on the 
        positive or negative profit.
        """
        ask_value = 0.0
        bid_value = 0.0
        sl_value = 0.0
        tp_value = 0.0
        is_profitable: bool = None

        # open position flag
        self.position_open = True

        # Check if the position is Short
        if action == "SELL":
            idx, tick_row = self.get_tick_row()
            if idx is None:
                # EOF at the very beginning
                print("No more ticks available — closing and stopping server.")
                self.position_open = False
                self.done = True
                return

            # read current ask and advance the pointer to next absolute row
            bid_value = float(tick_row["Bid price"])
            timestamp = tick_row.get("Timestamp", None)
            print(f"S row idx: {idx}, Timestamp: {timestamp}, Bid price: {bid_value}")

            # set pointer to next row (so next call starts after the one we just read)
            self.current_tick_row = idx + 1

            # compute SL/TP absolute prices
            tp_value = bid_value - (self.tp_pips * self.pip_decimal)
            sl_value = bid_value + (self.sl_pips * self.pip_decimal)

            # iterate subsequent ticks using absolute indexing
            for idx, row in self.iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                bid_price = float(row["Bid price"])
                ts = row.get("Timestamp", None)

                # update current pointer to this row (so external calls know where we are)
                self.current_tick_row = idx + 1  # next row to read on subsequent calls

                if bid_price <= tp_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Bid price: {bid_price} <= TP {tp_value} -> closing position (profit).")
                    self.position_open = False
                    is_profitable = True
                    break
                elif bid_price >= sl_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Bid price: {bid_price} > SL {sl_value} -> closing position (loss).")
                    self.position_open = False
                    is_profitable = False
                    break
                # else:
                    # still open — continue scanning (don't set position_open False here)
                    # print(f"Row {idx}, {ts}: Ask {ask_price} between SL and TP -> position still open.")
                    # continue loop until TP or SL or EOF

            if self.position_open:
                # EOF reached without hitting TP or SL
                print("Reached end of tick file before TP/SL. Closing position and stopping server.")
                self.position_open = False
                self.done = True

        # Check if the position is Long
        if action == "BUY":
            idx, tick_row = self.get_tick_row()
            if idx is None:
                # EOF at the very beginning
                print("No more ticks available — closing and stopping server.")
                self.position_open = False
                self.done = True
                return

            # read current ask and advance the pointer to next absolute row
            ask_value = float(tick_row["Ask price"])
            timestamp = tick_row.get("Timestamp", None)
            print(f"S row idx: {idx}, Timestamp: {timestamp}, Ask price: {ask_value}")

            # set pointer to next row (so next call starts after the one we just read)
            self.current_tick_row = idx + 1

            # compute SL/TP absolute prices
            tp_value = ask_value + (self.tp_pips * self.pip_decimal)
            sl_value = ask_value - (self.sl_pips * self.pip_decimal)

            # iterate subsequent ticks using absolute indexing
            for idx, row in self.iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                ask_price = float(row["Ask price"])
                ts = row.get("Timestamp", None)

                # update current pointer to this row (so external calls know where we are)
                self.current_tick_row = idx + 1  # next row to read on subsequent calls

                if ask_price >= tp_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Ask price: {ask_price} >= TP {tp_value} -> closing position (profit).")
                    self.position_open = False
                    is_profitable = True
                    break
                elif ask_price <= sl_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Ask price: {ask_price} <= SL {sl_value} -> closing position (loss).")
                    self.position_open = False
                    is_profitable = False
                    break
                # else:
                    # still open — continue scanning (don't set position_open False here)
                    # print(f"Row {idx}, {ts}: Ask {ask_price} between SL and TP -> position still open.")
                    # continue loop until TP or SL or EOF

            if self.position_open:
                # EOF reached without hitting TP or SL
                print("Reached end of tick file before TP/SL. Closing position and stopping server.")
                self.position_open = False
                self.done = True

        reward = self._calculate_reward(isProfitable=is_profitable)
        obs = self._get_observation(self.current_tick_row)
        info = {
            "position": "BUY" if action == "BUY" else "SELL",
            "tp": tp_value,
            "sl": sl_value,
            "final_idx": self.current_tick_row,
            "balance": self.balance,
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "max_drawdown": self.max_drawdown,
            "last_reward": reward
        }
        return obs, reward, self.done, self.truncated, info

    def _get_observation(self, idx):
        # read tick row
        for i, row in self.iter_ticks_from(start_row=idx, chunksize=1000):
            # only first row
            ask = float(row["Ask price"])
            bid = float(row["Bid price"])
            spread = ask - bid
            # later you can add indicators from your candles file
            obs = [ask, bid, spread, float(self.position_open)]
            return obs
        return [0.0, 0.0, 0.0, 0.0]

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

