from trading_env import TradingEnv

env = TradingEnv(pip_decimal=0.0001, candles_file="EURUSD_Daily.csv", tick_file="EURUSD_Ticks.csv")
obs, info = env.reset()
assert isinstance(obs, (list, tuple)) and len(obs) >= 4  # [ask,bid,spread,flag] 

obs, r, done, trunc, info = env.step("BUY")
assert isinstance(r, float) and isinstance(done, bool) and isinstance(trunc, bool)
assert "equity" in info and "max_drawdown" in info  # you added these in info, right? 

_ = env.reset()
_, r1, *_ = env.step("BUY")   # likely +pnl_norm, no DD penalty
_, r2, *_ = env.step("SELL")  # if loss → should include DD penalty
print("Rewards:", r1, r2, "MaxDD:", env.max_drawdown)  # MaxDD increases only when new valley is made

for a in ["BUY","SELL"]*3:
    _, r, _, _, i = env.step(a)
    print(f"{a}: r={r:.3f}, balance={i['balance']:.2f}, dd={i['max_drawdown']:.3f}")

