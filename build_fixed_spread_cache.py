import argparse
import os
import shutil

import numpy as np


def copy_if_exists(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache", default="cache_fx_EURUSD_D1")
    ap.add_argument("--dst_cache", default="cache_fx_EURUSD_D1_fx")
    ap.add_argument("--fixed_spread_points", type=float, default=18.0)
    ap.add_argument("--point_size", type=float, default=0.00001)
    args = ap.parse_args()

    os.makedirs(args.dst_cache, exist_ok=True)

    spread_price = float(args.fixed_spread_points) * float(args.point_size)

    # Copy artifacts that are unchanged
    for name in [
        "bars_features.npy",
        "bar_times.npy",
        "tick_to_bar.npy",
        "bars_mean.npy",
        "bars_std.npy",
        "meta.json",
    ]:
        copy_if_exists(os.path.join(args.src_cache, name), os.path.join(args.dst_cache, name))

    # Recompute ask from bid + fixed spread
    bid_path = os.path.join(args.src_cache, "tick_bid.npy")
    if not os.path.exists(bid_path):
        raise FileNotFoundError(f"Missing {bid_path}")

    tick_bid = np.load(bid_path, mmap_mode="r")
    tick_bid = np.asarray(tick_bid, dtype=np.float64)
    tick_ask = tick_bid + spread_price

    np.save(os.path.join(args.dst_cache, "tick_bid.npy"), tick_bid)
    np.save(os.path.join(args.dst_cache, "tick_ask.npy"), tick_ask)

    print("Built fixed-spread cache:")
    print("  src =", args.src_cache)
    print("  dst =", args.dst_cache)
    print("  fixed_spread_points =", args.fixed_spread_points)
    print("  point_size =", args.point_size)
    print("  spread_price =", spread_price)


if __name__ == "__main__":
    main()
