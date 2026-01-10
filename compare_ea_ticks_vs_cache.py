import argparse
from typing import Optional

import numpy as np
import pandas as pd


def read_csv_autosep(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] == 1 and df.columns.size == 1 and ";" in str(df.columns[0]):
            raise ValueError("autosep_failed_semicolon")
        return df
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_bar_time(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    s = s.str.replace("T", " ", regex=False)
    return pd.to_datetime(s, errors="coerce")


def read_ea_actions(path: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    bar_col = find_column(df, ["bar_index", "bar", "bar_idx"])
    if bar_col is None:
        raise KeyError(f"Missing bar_index in {path}. Columns={list(df.columns)}")

    time_col = find_column(df, ["bar_time", "bartime", "time", "datetime", "date"])
    bid_col = find_column(df, ["bid"])
    ask_col = find_column(df, ["ask"])

    cols = [bar_col]
    if time_col is not None:
        cols.append(time_col)
    if bid_col is not None:
        cols.append(bid_col)
    if ask_col is not None:
        cols.append(ask_col)

    out = df[cols].copy()
    out = out.rename(columns={bar_col: "bar_index"})
    out["bar_index"] = pd.to_numeric(out["bar_index"], errors="coerce")
    out = out.dropna(subset=["bar_index"]).copy()
    out["bar_index"] = out["bar_index"].astype(int)

    if time_col is not None:
        out = out.rename(columns={time_col: "bar_time"})
        out["bar_time"] = parse_bar_time(out["bar_time"])

    if bid_col is not None:
        out = out.rename(columns={bid_col: "bid"})
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    if ask_col is not None:
        out = out.rename(columns={ask_col: "ask"})
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")

    return out.dropna(subset=["bar_index"]).copy()


def read_cache(cache_dir: str) -> dict:
    tick_ask = np.load(f"{cache_dir}/tick_ask.npy", mmap_mode="r")
    tick_bid = np.load(f"{cache_dir}/tick_bid.npy", mmap_mode="r")
    tick_to_bar = np.load(f"{cache_dir}/tick_to_bar.npy", mmap_mode="r")
    bar_times = np.load(f"{cache_dir}/bar_times.npy", mmap_mode="r")

    bar_start_tick = np.searchsorted(tick_to_bar, np.arange(bar_times.size), side="left")

    return {
        "tick_ask": np.asarray(tick_ask),
        "tick_bid": np.asarray(tick_bid),
        "tick_to_bar": np.asarray(tick_to_bar),
        "bar_times": np.asarray(bar_times),
        "bar_start_tick": np.asarray(bar_start_tick),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_actions", default="ea_actions_2025.csv")
    ap.add_argument("--cache_dir", default="cache_fx_EURUSD_D1_fx")
    ap.add_argument("--pip_decimal", type=float, default=0.0001)
    ap.add_argument("--max_rows", type=int, default=10)
    args = ap.parse_args()

    ea = read_ea_actions(args.ea_actions)
    cache = read_cache(args.cache_dir)

    bar_idx = ea["bar_index"].to_numpy(np.int64)
    valid = (bar_idx >= 0) & (bar_idx < cache["bar_start_tick"].size)
    ea = ea.loc[valid].copy()
    bar_idx = ea["bar_index"].to_numpy(np.int64)

    start_ticks = cache["bar_start_tick"][bar_idx]
    cache_bid = cache["tick_bid"][start_ticks]
    cache_ask = cache["tick_ask"][start_ticks]

    ea["cache_bid"] = cache_bid
    ea["cache_ask"] = cache_ask

    if "bid" in ea.columns:
        ea["bid_diff"] = (ea["bid"] - ea["cache_bid"]).abs()
    if "ask" in ea.columns:
        ea["ask_diff"] = (ea["ask"] - ea["cache_ask"]).abs()

    cache_bar_time = pd.to_datetime(cache["bar_times"][bar_idx], unit="s", errors="coerce")
    ea["cache_bar_time"] = cache_bar_time

    if "bar_time" in ea.columns:
        ea["time_diff_sec"] = (ea["bar_time"] - ea["cache_bar_time"]).dt.total_seconds().abs()

    print("rows compared:", len(ea))
    if "bid_diff" in ea.columns:
        print("bid abs diff: mean=", ea["bid_diff"].mean(), "max=", ea["bid_diff"].max())
        print("bid diff (pips): mean=", ea["bid_diff"].mean() / args.pip_decimal,
              "max=", ea["bid_diff"].max() / args.pip_decimal)
    if "ask_diff" in ea.columns:
        print("ask abs diff: mean=", ea["ask_diff"].mean(), "max=", ea["ask_diff"].max())
        print("ask diff (pips): mean=", ea["ask_diff"].mean() / args.pip_decimal,
              "max=", ea["ask_diff"].max() / args.pip_decimal)
    if "time_diff_sec" in ea.columns:
        print("bar_time diff: mean_sec=", ea["time_diff_sec"].mean(),
              "max_sec=", ea["time_diff_sec"].max())

    # show top mismatches
    if "bid_diff" in ea.columns:
        top = ea.sort_values("bid_diff", ascending=False).head(args.max_rows)
        print("\nTop bid diffs:")
        print(top[["bar_index", "bar_time", "cache_bar_time", "bid", "cache_bid", "bid_diff"]].to_string(index=False))
    if "ask_diff" in ea.columns:
        top = ea.sort_values("ask_diff", ascending=False).head(args.max_rows)
        print("\nTop ask diffs:")
        print(top[["bar_index", "bar_time", "cache_bar_time", "ask", "cache_ask", "ask_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
