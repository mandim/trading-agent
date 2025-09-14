import pandas as pd

class TradingEnv:

    def __init__(self, candles_file: str, tick_file: str):
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.tick_line = 0
        self.position_open = False

    def open_position(self, position_name, sl_pips, tp_pips):
        
        ask_value = 0

        # Get the name and identify if it is a BUY or SELL position
        if position_name == "BUY":
            ask_value = self.tick_file
        
        # Get the current tick's ask or bid price depending of position and save it to a variable
        
        # Calculate the SL and TP by add or sub from the current price the SL/TP pips
        
        # Start comparing every tick_line each tick current price, increment the tick_line

        # If tick_line goes to the end of file then close the position
        
        # If the tick current price equals to TP/SL then close the posistion

        # Calculate profit

if __name__ == '__main__':

    env = TradingEnv('EURUSD_Daily.csv', 'EURUSD_ticks.csv')
    df = pd.read_csv(env.tick_file, nrows=100)
    print(df)
