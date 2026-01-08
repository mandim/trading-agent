import os
from datetime import datetime

import numpy as np
import pandas as pd


def load_actions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "bar_time" not in df.columns:
        raise KeyError(f"Missing bar_time in {path}")
    df["bar_time"] = (
        df["bar_time"].astype(str)
        .str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
        .str.replace("T", " ", regex=False)
    )
    df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce")
    return df.dropna(subset=["bar_time"])


def main():
    cache_dir = "cache_fx_EURUSD_D1"
    ea_actions = "ea_actions_2025.csv"
    test_actions = "test_actions_2025.csv"

    bar_times = np.load(os.path.join(cache_dir, "bar_times.npy"))
    bar_times_dt = pd.to_datetime(bar_times, unit="s")

    ea_df = load_actions(ea_actions)
    td_df = load_actions(test_actions)

    print("cache range:", bar_times_dt.min(), "->", bar_times_dt.max())
    print("ea range:", ea_df["bar_time"].min(), "->", ea_df["bar_time"].max())
    print("test range:", td_df["bar_time"].min(), "->", td_df["bar_time"].max())

    # Check exact alignment for a sample
    sample = ea_df["bar_time"].iloc[:5].to_list()
    print("sample EA times:", sample)

    # Verify each EA bar_time exists in cache (exact match)
    cache_set = set(bar_times_dt.to_pydatetime())
    missing = [t for t in ea_df["bar_time"] if t.to_pydatetime() not in cache_set]
    print("ea missing in cache:", len(missing))

    if missing:
        # Print the first few missing times to inspect offset
        print("first missing:", missing[:5])


if __name__ == "__main__":
    main()
