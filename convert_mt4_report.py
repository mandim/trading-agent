import pandas as pd
import os
import argparse

def convert_mt4_html_to_csv(html_path: str, out_csv: str):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    tables = pd.read_html(html)
    if len(tables) < 2:
        raise RuntimeError("Could not find MT4 trades table (expected at least 2 tables).")

    trade_raw = tables[1].copy()

    header = trade_raw.iloc[0].tolist()
    df = trade_raw.iloc[1:].copy()
    df.columns = header

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    for c in ["#", "Order"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Size", "Price", "S / L", "T / P", "Profit", "Balance"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="StrategyTester.htm")
    ap.add_argument("--out", default="mt4_trades_2025.csv")
    args = ap.parse_args()
    convert_mt4_html_to_csv(args.html, args.out)
