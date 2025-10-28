
import torch, numpy as np
from trading_env_gym import TradingEnv
from train_dqn import QNet

def load_and_eval(model_path="dqn_best.pt", episodes=5):
    env = TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir="cache_fx_EURUSD_D1",
        reward_mode="pnl",
        normalize_prices=True,
        eval_mode=True,
        max_steps_per_episode=5000,
        window_len=32,
        tp_pips=50.0,
        sl_pips=50.0,
        lot=1.0,
    )
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    net = QNet(obs_dim, n_actions)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    for ep in range(episodes):
        s, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done:
            with torch.no_grad():
                q = net(torch.from_numpy(s).float().unsqueeze(0))
                a = int(q.argmax(dim=1).item())
            s, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += r; steps += 1
        print(f"Episode {ep+1}: return={ep_ret:.3f}, steps={steps}")

if __name__ == "__main__":
    load_and_eval()
