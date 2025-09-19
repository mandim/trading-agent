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
        self.balance = 100000

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
            self.open_position("BUY", self.sl_pips, self.tp_pips)
            return {"reply": "BUY position closed and return current state to the Agent"}
        elif cmd == "SELL":
            self.open_position("SELL", self.sl_pips, self.tp_pips)
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

    def open_position(self, position_name: str, sl_pips: float, tp_pips: float):
        ask_value = 0.0
        bid_value = 0.0
        sl_value = 0.0
        tp_value = 0.0

        # open position flag
        self.position_open = True

        # Check if the position is Short
        if position_name == "SELL":
            idx, tick_row = self.get_tick_row()
            if idx is None:
                # EOF at the very beginning
                print("No more ticks available — closing and stopping server.")
                self.position_open = False
                self.stop_server()
                return

            # read current ask and advance the pointer to next absolute row
            bid_value = float(tick_row["Bid price"])
            timestamp = tick_row.get("Timestamp", None)
            print(f"S row idx: {idx}, Timestamp: {timestamp}, Bid price: {bid_value}")

            # set pointer to next row (so next call starts after the one we just read)
            self.current_tick_row = idx + 1

            # compute SL/TP absolute prices
            tp_value = bid_value - (tp_pips * self.pip_decimal)
            sl_value = bid_value + (sl_pips * self.pip_decimal)
            # print(f"TP value: {tp_value}")
            # print(f"SL value: {sl_value}")

            # iterate subsequent ticks using absolute indexing
            for idx, row in self.iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                bid_price = float(row["Bid price"])
                ts = row.get("Timestamp", None)

                # update current pointer to this row (so external calls know where we are)
                self.current_tick_row = idx + 1  # next row to read on subsequent calls

                if bid_price <= tp_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Bid price: {bid_price} <= TP {tp_value} -> closing position (profit).")
                    self.position_open = False
                    # Calculate new balance
                    self.balance = self.calculate_reward(self, True)
                    break
                elif bid_price > sl_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Bid price: {bid_price} > SL {sl_value} -> closing position (loss).")
                    self.position_open = False
                    break
                # else:
                    # still open — continue scanning (don't set position_open False here)
                    # print(f"Row {idx}, {ts}: Ask {ask_price} between SL and TP -> position still open.")
                    # continue loop until TP or SL or EOF

            if self.position_open:
                # EOF reached without hitting TP or SL
                print("Reached end of tick file before TP/SL. Closing position and stopping server.")
                self.position_open = False
                self.stop_server()

        # Check if the position is Long
        if position_name == "BUY":
            idx, tick_row = self.get_tick_row()
            if idx is None:
                # EOF at the very beginning
                print("No more ticks available — closing and stopping server.")
                self.position_open = False
                self.stop_server()
                return

            # read current ask and advance the pointer to next absolute row
            ask_value = float(tick_row["Ask price"])
            timestamp = tick_row.get("Timestamp", None)
            print(f"S row idx: {idx}, Timestamp: {timestamp}, Ask price: {ask_value}")

            # set pointer to next row (so next call starts after the one we just read)
            self.current_tick_row = idx + 1

            # compute SL/TP absolute prices
            tp_value = ask_value + (tp_pips * self.pip_decimal)
            sl_value = ask_value - (sl_pips * self.pip_decimal)
            # print(f"TP value: {tp_value}")
            # print(f"SL value: {sl_value}")

            # iterate subsequent ticks using absolute indexing
            for idx, row in self.iter_ticks_from(start_row=self.current_tick_row, chunksize=1000):
                ask_price = float(row["Ask price"])
                ts = row.get("Timestamp", None)

                # update current pointer to this row (so external calls know where we are)
                self.current_tick_row = idx + 1  # next row to read on subsequent calls

                if ask_price >= tp_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Ask price: {ask_price} >= TP {tp_value} -> closing position (profit).")
                    self.position_open = False
                    break
                elif ask_price <= sl_value:
                    print(f"E row idx: {idx}, Timestamp: {ts}, Ask price: {ask_price} <= SL {sl_value} -> closing position (loss).")
                    self.position_open = False
                    break
                else:
                    # still open — continue scanning (don't set position_open False here)
                    print(f"Row {idx}, {ts}: Ask {ask_price} between SL and TP -> position still open.")
                    # continue loop until TP or SL or EOF

            if self.position_open:
                # EOF reached without hitting TP or SL
                print("Reached end of tick file before TP/SL. Closing position and stopping server.")
                self.position_open = False
                self.stop_server()

        # TODO: calculate the next state and reward

    def calculate_reward(self, isProfitable: bool = None):
        
        old_balance = self.balance
        exchange_rate = 1 # Suppose that the quote currency is the same as the account currency
        pip_value = ((self.lot * 100000) * self.pip_decimal) / exchange_rate
        profit = 0
        
        if isProfitable:
            profit = pip_value * self.tp_pips
        elif isProfitable == False:
            profit = -pip_value * self.sl_pips
        
        next_balance = old_balance + profit
        self.balance = next_balance

        reward = math.log(next_balance / old_balance)

        return reward