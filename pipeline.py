"""
Market Intelligence Pipeline — Phase 1 (EOD)
=============================================
Downloads NSE's official daily bhavcopy (sec_bhavdata_full: price + delivery %),
maintains rolling per-stock history, computes indicators, runs scans,
and generates a plain-English narrative for every stock.

Usage:
  python pipeline.py daily              # fetch latest trading day (nightly job)
  python pipeline.py backfill 300       # first run: fetch last ~300 calendar days

Mock mode (for testing without internet):
  set env NSE_MOCK_DIR=/path/to/mock/csvs  (files named sec_bhavdata_full_DDMMYYYY.csv)

Outputs (all JSON, consumed by the dashboard):
  data/stocks/<SYMBOL>.json    per-stock: history, indicators, narrative
  data/scans.json              all scan results
  data/market.json             market overview (breadth, movers, sectors N/A in ph1)
  data/meta.json               last update info
"""
import io
import json
import os
import sys
import time
import zipfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(ROOT, "data", "history")
STOCK_DIR = os.path.join(ROOT, "data", "stocks")
DATA_DIR = os.path.join(ROOT, "data")
MAX_HISTORY_ROWS = 420          # ~20 months of trading days; enough for 200 EMA + 1Y returns
MIN_ROWS_FOR_INDICATORS = 30

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
      "Accept": "text/csv,application/csv,*/*"}

URL_TMPL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"


# ---------------------------------------------------------------- fetch
def fetch_day(d: date) -> pd.DataFrame | None:
    """Fetch one day's full bhavcopy (price + delivery). Returns None if unavailable."""
    tag = d.strftime("%d%m%Y")
    mock_dir = os.environ.get("NSE_MOCK_DIR")
    if mock_dir:
        path = os.path.join(mock_dir, f"sec_bhavdata_full_{tag}.csv")
        if not os.path.exists(path):
            return None
        raw = open(path, "rb").read()
    else:
        url = URL_TMPL.format(d=tag)
        for attempt in range(3):
            try:
                r = requests.get(url, headers=UA, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    raw = r.content
                    break
                if r.status_code == 404:
                    return None            # holiday / weekend / not published yet
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        else:
            return None

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None
    df.columns = [c.strip() for c in df.columns]
    # normalise expected columns
    need = ["SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
            "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_PER"]
    for c in need:
        if c not in df.columns:
            return None
    df["SERIES"] = df["SERIES"].astype(str).str.strip()
    df = df[df["SERIES"] == "EQ"].copy()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
    for c in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["DELIV_PER"] = pd.to_numeric(df["DELIV_PER"], errors="coerce")
    df = df.dropna(subset=["CLOSE_PRICE"])
    df["DATE"] = d.isoformat()
    return df[["SYMBOL", "DATE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
               "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_PER"]]


# ---------------------------------------------------------------- history store
def append_history(day_df: pd.DataFrame) -> int:
    os.makedirs(HIST_DIR, exist_ok=True)
    updated = 0
    for sym, g in day_df.groupby("SYMBOL"):
        if not sym.replace("-", "").replace("&", "").isalnum():
            continue
        path = os.path.join(HIST_DIR, f"{sym}.csv")
        row = g.iloc[[-1]]
        if os.path.exists(path):
            hist = pd.read_csv(path)
            if row["DATE"].iloc[0] in set(hist["DATE"]):
                continue
            hist = pd.concat([hist, row.drop(columns=["SYMBOL"])], ignore_index=True)
        else:
            hist = row.drop(columns=["SYMBOL"])
        hist = hist.sort_values("DATE").tail(MAX_HISTORY_ROWS)
        hist.to_csv(path, index=False)
        updated += 1
    return updated


# ---------------------------------------------------------------- indicators
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def macd(close):
    line = ema(close, 12) - ema(close, 26)
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig

def supertrend(df, period=10, mult=3.0):
    h, l, c = df["HIGH_PRICE"], df["LOW_PRICE"], df["CLOSE_PRICE"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    mid = (h + l) / 2
    ub, lb = mid + mult * atr, mid - mult * atr
    st = pd.Series(index=df.index, dtype=float)
    dirn = pd.Series(index=df.index, dtype=int)
    fu, fl = ub.iloc[0], lb.iloc[0]
    d = 1
    for i in range(len(df)):
        if i > 0:
            fu = min(ub.iloc[i], fu) if c.iloc[i - 1] <= fu else ub.iloc[i]
            fl = max(lb.iloc[i], fl) if c.iloc[i - 1] >= fl else lb.iloc[i]
            d = 1 if c.iloc[i] > fu and d == -1 else (-1 if c.iloc[i] < fl and d == 1 else d)
        st.iloc[i] = fl if d == 1 else fu
        dirn.iloc[i] = d
    return st, dirn

def compute_indicators(hist: pd.DataFrame) -> dict | None:
    if len(hist) < MIN_ROWS_FOR_INDICATORS:
        return None
    c = hist["CLOSE_PRICE"]
    out = {}
    out["close"] = round(float(c.iloc[-1]), 2)
    out["prev_close"] = round(float(c.iloc[-2]), 2) if len(c) > 1 else out["close"]
    out["chg_pct"] = round((out["close"] / out["prev_close"] - 1) * 100, 2) if out["prev_close"] else 0.0
    for n in (20, 50, 200):
        if len(c) >= n:
            out[f"ema{n}"] = round(float(ema(c, n).iloc[-1]), 2)
    r = rsi(c)
    out["rsi"] = round(float(r.iloc[-1]), 1) if not np.isnan(r.iloc[-1]) else None
    ml, ms = macd(c)
    out["macd"], out["macd_sig"] = round(float(ml.iloc[-1]), 2), round(float(ms.iloc[-1]), 2)
    bb_mid = c.rolling(20).mean()
    bb_sd = c.rolling(20).std()
    if not np.isnan(bb_mid.iloc[-1]):
        out["bb_up"] = round(float(bb_mid.iloc[-1] + 2 * bb_sd.iloc[-1]), 2)
        out["bb_dn"] = round(float(bb_mid.iloc[-1] - 2 * bb_sd.iloc[-1]), 2)
    st, dirn = supertrend(hist)
    out["supertrend"] = round(float(st.iloc[-1]), 2)
    out["st_dir"] = int(dirn.iloc[-1])
    out["st_flip"] = bool(len(dirn) > 1 and dirn.iloc[-1] != dirn.iloc[-2])
    lookbacks = {"ret_1w": 5, "ret_1m": 21, "ret_3m": 63, "ret_1y": 252}
    for k, n in lookbacks.items():
        if len(c) > n:
            out[k] = round((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100, 1)
    w = hist.tail(252)
    out["hi_52w"] = round(float(w["HIGH_PRICE"].max()), 2)
    out["lo_52w"] = round(float(w["LOW_PRICE"].min()), 2)
    out["from_hi_pct"] = round((out["close"] / out["hi_52w"] - 1) * 100, 1)
    out["from_lo_pct"] = round((out["close"] / out["lo_52w"] - 1) * 100, 1)
    v = hist["TTL_TRD_QNTY"]
    out["vol"] = int(v.iloc[-1])
    out["vol_avg20"] = int(v.tail(21).head(20).mean()) if len(v) > 20 else int(v.mean())
    out["vol_x"] = round(out["vol"] / out["vol_avg20"], 2) if out["vol_avg20"] else None
    dl = hist["DELIV_PER"].dropna()
    if len(dl) >= 21:
        out["deliv"] = round(float(dl.iloc[-1]), 1)
        out["deliv_avg20"] = round(float(dl.tail(21).head(20).mean()), 1)
    out["golden_cross"] = bool(
        "ema50" in out and "ema200" in out and len(c) >= 201 and
        float(ema(c, 50).iloc[-2]) <= float(ema(c, 200).iloc[-2]) and out["ema50"] > out["ema200"])
    return out


# ---------------------------------------------------------------- narrative engine
def narrative(sym: str, x: dict) -> str:
    s = []
    trend = []
    if x.get("ema200"):
        trend.append(f"{'above' if x['close'] > x['ema200'] else 'below'} its 200-day average")
    if x.get("ema50"):
        trend.append(f"{'above' if x['close'] > x['ema50'] else 'below'} the 50-day")
    if trend:
        s.append(f"{sym} closed at ₹{x['close']} ({x['chg_pct']:+.1f}%), trading {' and '.join(trend)}.")
    else:
        s.append(f"{sym} closed at ₹{x['close']} ({x['chg_pct']:+.1f}%).")
    if x.get("rsi") is not None:
        zone = "overbought territory" if x["rsi"] >= 70 else "oversold territory" if x["rsi"] <= 30 else None
        if zone:
            s.append(f"Momentum is stretched — RSI at {x['rsi']} is in {zone}.")
    if x.get("vol_x") and x["vol_x"] >= 1.8:
        d_note = ""
        if x.get("deliv") and x.get("deliv_avg20") and x["deliv"] > x["deliv_avg20"] + 5:
            d_note = f" with delivery at {x['deliv']}% vs a {x['deliv_avg20']}% average — a sign of genuine buying rather than churn"
        s.append(f"Today's volume ran {x['vol_x']}x its 20-day average{d_note}.")
    elif x.get("deliv") and x.get("deliv_avg20") and x["deliv"] > x["deliv_avg20"] + 8:
        s.append(f"Delivery share rose to {x['deliv']}% against a {x['deliv_avg20']}% norm, pointing to accumulation.")
    if x.get("st_flip"):
        s.append(f"Supertrend flipped {'bullish' if x['st_dir'] == 1 else 'bearish'} today.")
    if x.get("golden_cross"):
        s.append("The 50-day average crossed above the 200-day — a golden cross.")
    if x.get("from_hi_pct") is not None and x["from_hi_pct"] >= -1:
        s.append("Price is at a fresh 52-week high.")
    elif x.get("from_hi_pct") is not None and x["from_hi_pct"] > -8:
        s.append(f"Price sits {abs(x['from_hi_pct'])}% below its 52-week high.")
    if x.get("ret_1m") is not None:
        s.append(f"Over the past month it is {x['ret_1m']:+.1f}%; over a year {x.get('ret_1y', 0):+.1f}%.")
    return " ".join(s)


# ---------------------------------------------------------------- scans + market
SCAN_DEFS = {
    "above_200ema":   ("Trading above 200-day EMA",        lambda x: x.get("ema200") and x["close"] > x["ema200"]),
    "rsi_oversold":   ("RSI below 30 (oversold)",          lambda x: x.get("rsi") is not None and x["rsi"] < 30),
    "rsi_overbought": ("RSI above 70 (overbought)",        lambda x: x.get("rsi") is not None and x["rsi"] > 70),
    "st_flip_bull":   ("Supertrend flipped bullish today", lambda x: x.get("st_flip") and x["st_dir"] == 1),
    "st_flip_bear":   ("Supertrend flipped bearish today", lambda x: x.get("st_flip") and x["st_dir"] == -1),
    "vol_spike":      ("Volume 2x+ its 20-day average",    lambda x: x.get("vol_x") and x["vol_x"] >= 2),
    "deliv_accum":    ("Rising delivery % + rising price", lambda x: x.get("deliv") and x.get("deliv_avg20")
                                                            and x["deliv"] > x["deliv_avg20"] + 5 and x["chg_pct"] > 0),
    "hi_52w":         ("At / within 1% of 52-week high",   lambda x: x.get("from_hi_pct") is not None and x["from_hi_pct"] >= -1),
    "golden_cross":   ("Golden cross (50>200) today",      lambda x: x.get("golden_cross")),
}


def rebuild_outputs():
    os.makedirs(STOCK_DIR, exist_ok=True)
    scans = {k: [] for k in SCAN_DEFS}
    universe, latest_date = [], None
    for f in sorted(os.listdir(HIST_DIR)):
        if not f.endswith(".csv"):
            continue
        sym = f[:-4]
        hist = pd.read_csv(os.path.join(HIST_DIR, f))
        ind = compute_indicators(hist)
        if ind is None:
            continue
        latest_date = max(latest_date or "", hist["DATE"].iloc[-1])
        ind["narrative"] = narrative(sym, ind)
        tail = hist.tail(260)
        payload = {"symbol": sym, "asof": hist["DATE"].iloc[-1], "ind": ind,
                   "series": {"date": tail["DATE"].tolist(),
                              "o": tail["OPEN_PRICE"].round(2).tolist(),
                              "h": tail["HIGH_PRICE"].round(2).tolist(),
                              "l": tail["LOW_PRICE"].round(2).tolist(),
                              "c": tail["CLOSE_PRICE"].round(2).tolist(),
                              "v": tail["TTL_TRD_QNTY"].astype(int).tolist(),
                              "dlv": tail["DELIV_PER"].round(1).fillna(0).tolist()}}
        with open(os.path.join(STOCK_DIR, f"{sym}.json"), "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        row = {"symbol": sym, "close": ind["close"], "chg": ind["chg_pct"],
               "rsi": ind.get("rsi"), "vol_x": ind.get("vol_x"),
               "deliv": ind.get("deliv"), "ret_1m": ind.get("ret_1m")}
        universe.append(row)
        for k, (_, fn) in SCAN_DEFS.items():
            try:
                if fn(ind):
                    scans[k].append(row)
            except Exception:
                pass

    u = pd.DataFrame(universe)
    market = {}
    if len(u):
        market = {
            "asof": latest_date, "stocks": int(len(u)),
            "advances": int((u["chg"] > 0).sum()), "declines": int((u["chg"] < 0).sum()),
            "unchanged": int((u["chg"] == 0).sum()),
            "top_gainers": u.nlargest(10, "chg").to_dict("records"),
            "top_losers": u.nsmallest(10, "chg").to_dict("records"),
            "most_active_volx": u.dropna(subset=["vol_x"]).nlargest(10, "vol_x").to_dict("records"),
            "highest_delivery": u.dropna(subset=["deliv"]).nlargest(10, "deliv").to_dict("records"),
        }
    with open(os.path.join(DATA_DIR, "scans.json"), "w") as fh:
        json.dump({"asof": latest_date,
                   "scans": [{"id": k, "name": SCAN_DEFS[k][0], "count": len(v), "rows": v[:100]}
                             for k, v in scans.items()]}, fh, separators=(",", ":"))
    with open(os.path.join(DATA_DIR, "market.json"), "w") as fh:
        json.dump(market, fh, separators=(",", ":"))
    with open(os.path.join(DATA_DIR, "symbols.json"), "w") as fh:
        json.dump(sorted(u["symbol"].tolist()) if len(u) else [], fh)
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as fh:
        json.dump({"updated_utc": datetime.utcnow().isoformat() + "Z",
                   "asof": latest_date, "stocks": len(universe)}, fh)
    return len(universe), latest_date


# ---------------------------------------------------------------- entrypoints
def run_daily():
    d = date.today()
    for _ in range(7):                      # walk back over weekends/holidays
        df = fetch_day(d)
        if df is not None and len(df):
            n = append_history(df)
            print(f"fetched {d} rows={len(df)} new_history_rows={n}")
            break
        d -= timedelta(days=1)
    else:
        print("no trading day found in the last 7 days — nothing to do")
    n, asof = rebuild_outputs()
    print(f"outputs rebuilt: {n} stocks, as of {asof}")


def run_backfill(days: int):
    got = 0
    for i in range(days, -1, -1):
        d = date.today() - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        df = fetch_day(d)
        if df is None:
            continue
        append_history(df)
        got += 1
        if got % 20 == 0:
            print(f"...{got} trading days ingested (latest {d})")
        if not os.environ.get("NSE_MOCK_DIR"):
            time.sleep(0.6)                 # be polite to NSE
    print(f"backfill complete: {got} trading days")
    n, asof = rebuild_outputs()
    print(f"outputs rebuilt: {n} stocks, as of {asof}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "backfill":
        run_backfill(int(sys.argv[2]) if len(sys.argv) > 2 else 320)
    else:
        run_daily()
