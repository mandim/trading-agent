import pandas as pd
from server import ZMQRepServer

class TradingEnv:

    def __init__(self, pip_value: float, candles_file: str, tick_file: str, bind_address="tcp://*:5555"):
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.current_tick = 0
        self.position_open = False
        self.server = ZMQRepServer(bind_address, self.handle_request)
        self.counter = 0
        self.pip_value = pip_value

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
        self.counter += 1
        # Simple example: support {"cmd": "ping"} and {"cmd": "sum", "values": [..]}
        cmd = request.get("cmd") if isinstance(request, dict) else None

        if cmd == "BUY":
            self.open_position("BUY", 50, 100)
            return {"reply": "BUY position opened", "count": self.counter}
        elif cmd == "sum":
            vals = request.get("values", [])
            try:
                s = sum(vals)
                return {"reply": "ok", "sum": s}
            except Exception as e:
                return {"reply": "error", "error": str(e)}
        else:
            # default: echo
            return {"reply": "unknown_command", "received": request}

    def get_tick(self):
        tick_row = None        
        for chunk in pd.read_csv(self.tick_file, chunksize=1000):
            if self.current_tick in chunk.index:
                tick_row = chunk.loc[self.current_tick]
                break
        return tick_row

    def open_position(self, position_name: str, sl_pips: float, tp_pips: float):
        
        ask_value = 0
        bid_value = 0
        sl_value = 0
        tp_value = 0

        self.position_open = True

        # Get the name and identify if it is a BUY or SELL position
        if position_name == "BUY":
            
            # Get the ask price of the current position in ticks file
            tick_row = self.get_tick()
            if tick_row.__bool__:
                ask_value = tick_row["Ask price"]
                print(f"Ask price: {ask_value}")
                self.current_tick+=1
            
            # Calculate the SL and TP by add or sub from the current price the SL/TP pips
            tp_value = ask_value + (tp_pips * self.pip_value)
            sl_value = ask_value - (sl_pips * self.pip_value)
            print(f"TP value: {tp_value}")
            print(f"SL value: {sl_value}")
        
        
        
        # Start comparing every tick_line each tick current price, increment the tick_line

        # If tick_line goes to the end of file then close the position
        
        # If the tick current price equals to TP/SL then close the posistion

        # Calculate profit
