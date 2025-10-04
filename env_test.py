# env_test.py — refactored
import numpy as np
from trading_env import TradingEnv

def make_env():
    # cache_dir uses the env default unless you want to pass a specific path
    return TradingEnv(
        pip_decimal=0.0001,
        candles_file="EURUSD_Daily.csv",
        tick_file="EURUSD_Ticks.csv",
        # cache_dir="cache_fx_EURUSD_D1",  # uncomment if you named your cache differently
    )

def test_reset_and_observation_shape():
    env = make_env()
    obs, info = env.reset()

    # Expect NumPy vector: window_len * n_feats + 4 ([ask,bid,spread,position_open])
    assert isinstance(obs, np.ndarray), "obs should be a NumPy array"
    expected_len = env.window_len * env.n_feats + 4
    assert obs.ndim == 1 and obs.shape[0] == expected_len, \
        f"obs length {obs.shape[0]} != expected {expected_len}"
    assert obs.dtype == np.float32, "obs dtype should be float32"
    # info is a dict
    assert isinstance(info, dict)

def test_step_returns_and_info_keys():
    env = make_env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step("BUY")

    # Types
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Useful diagnostics present
    for key in ("equity", "max_drawdown", "action", "final_idx"):
        assert key in info, f"missing '{key}' in info"

def test_index_monotonic_and_rewards_progression():
    env = make_env()
    env.reset()

    idx_prev = None
    rewards = []
    actions = ["BUY", "SELL", "HOLD", "BUY", "SELL", "HOLD"]

    for a in actions:
        _, r, term, trunc, i = env.step(a)
        rewards.append(float(r))
        assert "final_idx" in i
        if idx_prev is not None:
            assert i["final_idx"] >= idx_prev, "tick index should be non-decreasing"
        idx_prev = i["final_idx"]
        if term:
            break

    # Basic sanity: at least one reward collected
    assert len(rewards) >= 1

if __name__ == "__main__":
    # Allow running as a simple script, similar to your original test
    env = make_env()
    obs, info = env.reset()
    print("Obs shape:", obs.shape, "dtype:", obs.dtype)

    obs, r, done, trunc, info = env.step("BUY")
    print("Step BUY -> r:", r, "done:", done, "trunc:", trunc, "dd:", info.get("max_drawdown"))

    env.reset()
    _, r1, *_ = env.step("BUY")
    _, r2, *_ = env.step("SELL")
    print("Rewards:", r1, r2, "MaxDD:", getattr(env, "max_drawdown", None))

    for a in ["BUY", "SELL"] * 3:
        _, r, d, t, i = env.step(a)
        print(f"{a}: r={r:.3f}, balance={i.get('balance', None)}, dd={i.get('max_drawdown', None)}")
        if d:
            break
