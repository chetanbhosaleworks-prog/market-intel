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

    df = pd.read_csv(io.BytesIO(raw))
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
    # --- pattern & level flags (EOD, based on last bars) ---
    o_, h_, l_, c_ = (hist["OPEN_PRICE"].values, hist["HIGH_PRICE"].values,
                      hist["LOW_PRICE"].values, hist["CLOSE_PRICE"].values)
    n = len(c_)
    def body(i): return abs(c_[i] - o_[i])
    def bull(i): return c_[i] > o_[i]
    def bear(i): return c_[i] < o_[i]
    pat = []
    if n >= 2:
        b1, b0 = body(-2), body(-1)
        if bear(-2) and bull(-1) and o_[-1] <= c_[-2] and c_[-1] >= o_[-2] and b0 > b1:
            pat.append("bull_engulf")
        if bull(-2) and bear(-1) and o_[-1] >= c_[-2] and c_[-1] <= o_[-2] and b0 > b1:
            pat.append("bear_engulf")
        mid_prev = (o_[-2] + c_[-2]) / 2
        if bear(-2) and bull(-1) and o_[-1] < c_[-2] and c_[-1] > mid_prev and c_[-1] < o_[-2]:
            pat.append("piercing")
        if bull(-2) and bear(-1) and o_[-1] > c_[-2] and c_[-1] < mid_prev and c_[-1] > o_[-2]:
            pat.append("dark_cloud")
    if n >= 6:
        rng = h_[-1] - l_[-1]
        bd = body(-1)
        low_sh = min(o_[-1], c_[-1]) - l_[-1]
        up_sh = h_[-1] - max(o_[-1], c_[-1])
        if rng > 0 and bd > 0:
            if low_sh >= 2 * bd and up_sh <= 0.5 * bd and l_[-1] <= min(l_[-6:-1]):
                pat.append("hammer")
            if up_sh >= 2 * bd and low_sh <= 0.5 * bd and h_[-1] >= max(h_[-6:-1]):
                pat.append("shooting_star")
    if n >= 4:
        if all(bull(i) for i in (-3, -2, -1)) and c_[-1] > c_[-2] > c_[-3]            and o_[-1] > o_[-2] and o_[-2] > o_[-3]:
            pat.append("three_soldiers")
        if all(bear(i) for i in (-3, -2, -1)) and c_[-1] < c_[-2] < c_[-3]            and o_[-1] < o_[-2] and o_[-2] < o_[-3]:
            pat.append("three_crows")
    out["patterns"] = pat
    if n >= 3:
        pH, pL, pC = h_[-2], l_[-2], c_[-2]
        P = (pH + pL + pC) / 3
        lv = {"r1": 2 * P - pL, "r2": P + (pH - pL), "r3": pH + 2 * (P - pL),
              "s1": 2 * P - pH, "s2": P - (pH - pL), "s3": pL - 2 * (pH - P)}
        for k in ("r1", "r2", "r3"):
            out["x_" + k] = bool(c_[-2] < lv[k] <= c_[-1] or (c_[-2] <= lv[k] < c_[-1]))
        for k in ("s1", "s2", "s3"):
            out["x_" + k] = bool(c_[-2] > lv[k] >= c_[-1] or (c_[-2] >= lv[k] > c_[-1]))
    if n >= 8:
        ranges = (h_ - l_)[-7:]
        out["nr7"] = bool(ranges[-1] > 0 and ranges[-1] <= ranges.min())
    if n >= 7:
        out["wk1_hi"] = bool(h_[-1] >= max(h_[-6:-1]))
        out["wk1_lo"] = bool(l_[-1] <= min(l_[-6:-1]))
    if n >= 22:
        out["wk4_hi"] = bool(h_[-1] >= max(h_[-21:-1]))
        out["wk4_lo"] = bool(l_[-1] <= min(l_[-21:-1]))
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




# ---------------------------------------------------------------- indices (NSE official EOD)
IDX_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d}.csv"
IDX_WANT = {"Nifty 50": "NIFTY 50", "Nifty Bank": "BANK NIFTY",
            "Nifty Financial Services": "FIN NIFTY",
            "Nifty Midcap Select": "MIDCAP SELECT", "Nifty 500": "NIFTY 500",
            "India VIX": "INDIA VIX"}
SECT_WANT = {"Nifty IT": "IT", "Nifty Pharma": "PHARMA", "Nifty Auto": "AUTO",
             "Nifty Bank": "BANK", "Nifty FMCG": "FMCG", "Nifty Metal": "METAL",
             "Nifty Realty": "REALTY", "Nifty Energy": "ENERGY",
             "Nifty PSU Bank": "PSU BANK", "Nifty Media": "MEDIA",
             "Nifty Private Bank": "PVT BANK", "Nifty Consumer Durables": "CONS DUR",
             "Nifty Oil & Gas": "OIL & GAS", "Nifty Healthcare Index": "HEALTHCARE",
             "Nifty Infrastructure": "INFRA"}
SECT_HIST = os.path.join(ROOT, "data", "history_sect")
IDX_HIST = os.path.join(ROOT, "data", "history_idx")

def fetch_indices(d: date) -> pd.DataFrame | None:
    tag = d.strftime("%d%m%Y")
    mock_dir = os.environ.get("NSE_MOCK_DIR")
    if mock_dir:
        p = os.path.join(mock_dir, f"ind_close_all_{tag}.csv")
        if not os.path.exists(p):
            return None
        raw = open(p, "rb").read()
    else:
        for attempt in range(3):
            try:
                r = requests.get(IDX_URL.format(d=tag), headers=UA, timeout=30)
                if r.status_code == 200 and len(r.content) > 500:
                    raw = r.content
                    break
                if r.status_code == 404:
                    return None
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
    name_c = next((c for c in df.columns if c.lower().startswith("index name")), None)
    close_c = next((c for c in df.columns if c.lower().startswith("closing")), None)
    pe_c = next((c for c in df.columns if c.strip().lower() in ("p/e", "pe")), None)
    pb_c = next((c for c in df.columns if c.strip().lower() in ("p/b", "pb")), None)
    if not name_c or not close_c:
        return None
    def norm(x):
        return " ".join(str(x).strip().lower().split())
    lut = {}
    for k in list(IDX_WANT.keys()) + list(SECT_WANT.keys()):
        lut[norm(k)] = k
    df["_N"] = df[name_c].map(lambda v: lut.get(norm(v)))
    df = df[df["_N"].notna()].copy()
    df["DATE"] = d.isoformat()
    o_c = next((c for c in df.columns if c.lower().startswith("open index")), None)
    h_c = next((c for c in df.columns if c.lower().startswith("high index")), None)
    l_c = next((c for c in df.columns if c.lower().startswith("low index")), None)
    cols = {"_N": "NAME", close_c: "CLOSE"}
    keep_c = ["_N", close_c, "DATE"]
    for src, dst in ((o_c, "OPEN"), (h_c, "HIGH"), (l_c, "LOW")):
        if src:
            cols[src] = dst; keep_c.insert(2, src)
    if pe_c:
        cols[pe_c] = "PE"; keep_c.insert(2, pe_c)
    if pb_c:
        cols[pb_c] = "PB"; keep_c.insert(3, pb_c)
    df = df[keep_c].rename(columns=cols)
    for cc in ("CLOSE", "PE", "PB"):
        if cc in df.columns:
            df[cc] = pd.to_numeric(df[cc], errors="coerce")
    return df.dropna(subset=["CLOSE"])

def append_indices(day_df: pd.DataFrame):
    os.makedirs(IDX_HIST, exist_ok=True)
    os.makedirs(SECT_HIST, exist_ok=True)
    for _, r in day_df.iterrows():
        targets = []
        if r["NAME"] in IDX_WANT:
            targets.append((IDX_HIST, IDX_WANT[r["NAME"]]))
        if r["NAME"] in SECT_WANT:
            targets.append((SECT_HIST, SECT_WANT[r["NAME"]]))
        for hist_dir, label in targets:
            _append_one_index(hist_dir, label, r)

def _append_one_index(hist_dir, label, r):
    key = label.replace(" ", "_").replace("&", "and")
    p = os.path.join(hist_dir, key + ".csv")
    rec = {"DATE": r["DATE"], "CLOSE": float(r["CLOSE"])}
    for c in ("OPEN", "HIGH", "LOW"):
        if c in r.index and not pd.isna(r[c]):
            rec[c] = float(r[c])
    row = pd.DataFrame([rec])
    if os.path.exists(p):
        h = pd.read_csv(p)
        if r["DATE"] in set(h["DATE"]):
            return
        h = pd.concat([h, row], ignore_index=True)
    else:
        h = row
    h.sort_values("DATE").tail(420).to_csv(p, index=False)

def build_indices_json():
    out = []
    if not os.path.isdir(IDX_HIST):
        return
    for name, label in IDX_WANT.items():
        p = os.path.join(IDX_HIST, label.replace(" ", "_").replace("&", "and") + ".csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        if len(h) < 2:
            continue
        c, pv = float(h["CLOSE"].iloc[-1]), float(h["CLOSE"].iloc[-2])
        out.append({"label": label, "close": round(c, 2),
                    "chg": round(c - pv, 2), "chg_pct": round((c / pv - 1) * 100, 2),
                    "spark": [round(x, 1) for x in h["CLOSE"].tail(30).tolist()],
                    "asof": h["DATE"].iloc[-1]})
    with open(os.path.join(DATA_DIR, "indices.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

VAL_HIST = os.path.join(ROOT, "data", "history_val")
VAL_WANT = {"Nifty 50": "NIFTY 50", "Nifty 500": "NIFTY 500"}

def append_valuation(day_df: pd.DataFrame):
    if "PE" not in day_df.columns:
        return
    os.makedirs(VAL_HIST, exist_ok=True)
    for name, label in VAL_WANT.items():
        r = day_df[day_df["NAME"] == name]
        if not len(r) or pd.isna(r["PE"].iloc[0]):
            continue
        p = os.path.join(VAL_HIST, label.replace(" ", "_") + ".csv")
        row = pd.DataFrame([{"DATE": r["DATE"].iloc[0], "PE": float(r["PE"].iloc[0]),
                             "PB": float(r["PB"].iloc[0]) if "PB" in r.columns and not pd.isna(r["PB"].iloc[0]) else None}])
        if os.path.exists(p):
            h = pd.read_csv(p)
            if row["DATE"].iloc[0] in set(h["DATE"]):
                continue
            h = pd.concat([h, row], ignore_index=True)
        else:
            h = row
        h.sort_values("DATE").tail(90).to_csv(p, index=False)

def build_valuation_json():
    out = {}
    if not os.path.isdir(VAL_HIST):
        return
    for name, label in VAL_WANT.items():
        p = os.path.join(VAL_HIST, label.replace(" ", "_") + ".csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        if not len(h):
            continue
        out[label] = {"pe": round(float(h["PE"].iloc[-1]), 2),
                      "pb": round(float(h["PB"].iloc[-1]), 2) if "PB" in h.columns and not pd.isna(h["PB"].iloc[-1]) else None,
                      "spark": [round(x, 2) for x in h["PE"].tail(30).tolist()],
                      "asof": h["DATE"].iloc[-1]}
    if out:
        with open(os.path.join(DATA_DIR, "valuation.json"), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))

def build_sectors_json():
    out = []
    if not os.path.isdir(SECT_HIST):
        return
    for name, label in SECT_WANT.items():
        p = os.path.join(SECT_HIST, label.replace(" ", "_").replace("&", "and") + ".csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        if len(h) < 2:
            continue
        c, pv = float(h["CLOSE"].iloc[-1]), float(h["CLOSE"].iloc[-2])
        w = h["CLOSE"].tail(6)
        wk = round((c / float(w.iloc[0]) - 1) * 100, 2) if len(w) >= 6 else None
        out.append({"label": label, "close": round(c, 2),
                    "chg_pct": round((c / pv - 1) * 100, 2), "wk_pct": wk,
                    "asof": h["DATE"].iloc[-1]})
    out.sort(key=lambda r: -r["chg_pct"])
    with open(os.path.join(DATA_DIR, "sectors.json"), "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

EQ_MASTER_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
N50_URLS = ["https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
            "https://niftyindices.com/IndexConstituent/ind_nifty50list.csv"]

def _fetch_csv(url, mock_name):
    mock_dir = os.environ.get("NSE_MOCK_DIR")
    if mock_dir:
        p = os.path.join(mock_dir, mock_name)
        if not os.path.exists(p):
            return None
        return open(p, "rb").read()
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            if r.status_code == 200 and len(r.content) > 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None

def build_names_json():
    raw = _fetch_csv(EQ_MASTER_URL, "EQUITY_L.csv")
    if raw is None:
        return
    try:
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip() for c in df.columns]
        sym_c = next((c for c in df.columns if c.upper() == "SYMBOL"), None)
        nm_c = next((c for c in df.columns if "NAME OF COMPANY" in c.upper()), None)
        if not sym_c or not nm_c:
            return
        out = {str(r[sym_c]).strip(): str(r[nm_c]).strip()
               for _, r in df.iterrows() if str(r[sym_c]).strip()}
        with open(os.path.join(DATA_DIR, "names.json"), "w") as fh:
            json.dump(out, fh, separators=(",", ":"))
        print(f"names master: {len(out)} companies")
    except Exception as e:
        print(f"names master skipped: {type(e).__name__}")

def build_n50_map():
    raw = None
    for u in N50_URLS:
        raw = _fetch_csv(u, "ind_nifty50list.csv")
        if raw is not None:
            break
    syms = []
    if raw is not None:
        try:
            df = pd.read_csv(io.BytesIO(raw))
            df.columns = [c.strip() for c in df.columns]
            sc = next((c for c in df.columns if c.upper() == "SYMBOL"), None)
            if sc:
                syms = [str(x).strip() for x in df[sc].tolist()]
        except Exception:
            pass
    if not syms:
        return
    with open(os.path.join(DATA_DIR, "constituents.json"), "w") as fh:
        json.dump({"NIFTY 50": syms}, fh, separators=(",", ":"))
    rows = []
    for sym in syms:
        p = os.path.join(HIST_DIR, f"{sym}.csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p).tail(2)
        if len(h) < 2:
            continue
        c, pv = float(h["CLOSE_PRICE"].iloc[-1]), float(h["CLOSE_PRICE"].iloc[-2])
        rows.append({"symbol": sym, "close": round(c, 2),
                     "chg": round((c / pv - 1) * 100, 2)})
    with open(os.path.join(DATA_DIR, "n50map.json"), "w") as fh:
        json.dump(sorted(rows, key=lambda r: -r["chg"]), fh, separators=(",", ":"))
    print(f"nifty50 map: {len(rows)} stocks")

def build_holidays_json():
    p = os.path.join(IDX_HIST, "NIFTY_50.csv")
    if not os.path.exists(p):
        return
    h = pd.read_csv(p)
    have = set(h["DATE"])
    if len(have) < 10:
        return
    dts = sorted(have)
    start = datetime.strptime(dts[0], "%Y-%m-%d").date()
    end = datetime.strptime(dts[-1], "%Y-%m-%d").date()
    closed = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d.isoformat() not in have:
            closed.append(d.isoformat())
        d += timedelta(days=1)
    with open(os.path.join(DATA_DIR, "holidays.json"), "w") as fh:
        json.dump({"observed_closures": closed[-25:],
                   "note": "Weekdays with no NSE trading data in our records — observed market closures."},
                  fh, separators=(",", ":"))

IDXPAGE_DIR = os.path.join(DATA_DIR, "indexpages")

def build_index_pages():
    os.makedirs(IDXPAGE_DIR, exist_ok=True)
    for name, label in IDX_WANT.items():
        p = os.path.join(IDX_HIST, label.replace(" ", "_").replace("&", "and") + ".csv")
        if not os.path.exists(p):
            continue
        h = pd.read_csv(p)
        if len(h) < 5:
            continue
        c = h["CLOSE"]
        x = {"close": round(float(c.iloc[-1]), 2),
             "prev_close": round(float(c.iloc[-2]), 2)}
        x["chg_pct"] = round((x["close"] / x["prev_close"] - 1) * 100, 2) if x["prev_close"] else 0.0
        for n in (20, 50, 200):
            if len(c) >= n:
                x[f"ema{n}"] = round(float(ema(c, n).iloc[-1]), 2)
        r = rsi(c)
        x["rsi"] = round(float(r.iloc[-1]), 1) if len(c) > 15 and not np.isnan(r.iloc[-1]) else None
        for k, n in {"ret_1w": 5, "ret_1m": 21, "ret_3m": 63, "ret_1y": 252}.items():
            if len(c) > n:
                x[k] = round((c.iloc[-1] / c.iloc[-1 - n] - 1) * 100, 1)
        hi_src = h["HIGH"] if "HIGH" in h.columns and h["HIGH"].notna().sum() > 5 else c
        lo_src = h["LOW"] if "LOW" in h.columns and h["LOW"].notna().sum() > 5 else c
        w = min(len(h), 252)
        x["hi_52w"] = round(float(hi_src.tail(w).max()), 2)
        x["lo_52w"] = round(float(lo_src.tail(w).min()), 2)
        x["hi_1m"] = round(float(hi_src.tail(21).max()), 2)
        x["lo_1m"] = round(float(lo_src.tail(21).min()), 2)
        x["hi_1w"] = round(float(hi_src.tail(5).max()), 2)
        x["lo_1w"] = round(float(lo_src.tail(5).min()), 2)
        x["from_hi_pct"] = round((x["close"] / x["hi_52w"] - 1) * 100, 1)
        x["from_lo_pct"] = round((x["close"] / x["lo_52w"] - 1) * 100, 1)
        bulls = sum([1 if x.get("ema20") and x["close"] > x["ema20"] else 0,
                     1 if x.get("ema50") and x["close"] > x["ema50"] else 0,
                     1 if x.get("ema200") and x["close"] > x["ema200"] else 0,
                     1 if (x.get("rsi") or 0) > 50 else 0])
        x["verdict"] = "Bullish" if bulls >= 3 else "Bearish" if bulls <= 1 else "Mixed"
        tail = h.tail(260)
        series = {"date": tail["DATE"].tolist(), "c": tail["CLOSE"].round(2).tolist()}
        for col, k in (("OPEN", "o"), ("HIGH", "h"), ("LOW", "l")):
            if col in tail.columns and tail[col].notna().sum() > 0:
                series[k] = [None if pd.isna(v) else round(float(v), 2) for v in tail[col]]
        with open(os.path.join(IDXPAGE_DIR, label.replace(" ", "_").replace("&", "and") + ".json"), "w") as fh:
            json.dump({"label": label, "asof": h["DATE"].iloc[-1], "ind": x, "series": series},
                      fh, separators=(",", ":"))

# ---------------------------------------------------------------- FII / participant-wise F&O OI
FII_URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d}.csv"
FII_HIST = os.path.join(ROOT, "data", "history_fii")
FII_FILE = os.path.join(FII_HIST, "participant_oi.csv")

def fetch_fii(d: date) -> pd.DataFrame | None:
    tag = d.strftime("%d%m%Y")
    mock_dir = os.environ.get("NSE_MOCK_DIR")
    if mock_dir:
        p = os.path.join(mock_dir, f"fao_participant_oi_{tag}.csv")
        if not os.path.exists(p):
            return None
        raw = open(p, "rb").read()
    else:
        for attempt in range(3):
            try:
                r = requests.get(FII_URL.format(d=tag), headers=UA, timeout=30)
                if r.status_code == 200 and len(r.content) > 300:
                    raw = r.content
                    break
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        else:
            return None
    for skip in (0, 1):
        try:
            df = pd.read_csv(io.BytesIO(raw), skiprows=skip)
            df.columns = [str(c).strip() for c in df.columns]
            ct = next((c for c in df.columns if c.lower().startswith("client type")), None)
            fl = next((c for c in df.columns if c.lower().startswith("future index long")), None)
            fs = next((c for c in df.columns if c.lower().startswith("future index short")), None)
            if ct and fl and fs:
                df[ct] = df[ct].astype(str).str.strip()
                df = df[df[ct].isin(["FII", "DII", "Client", "Pro"])]
                out = df[[ct, fl, fs]].rename(columns={ct: "WHO", fl: "LONG", fs: "SHORT"})
                out["LONG"] = pd.to_numeric(out["LONG"], errors="coerce")
                out["SHORT"] = pd.to_numeric(out["SHORT"], errors="coerce")
                out["DATE"] = d.isoformat()
                return out.dropna()
        except Exception:
            continue
    return None

def append_fii(day_df: pd.DataFrame):
    os.makedirs(FII_HIST, exist_ok=True)
    if os.path.exists(FII_FILE):
        h = pd.read_csv(FII_FILE)
        if day_df["DATE"].iloc[0] in set(h["DATE"]):
            return
        h = pd.concat([h, day_df], ignore_index=True)
    else:
        h = day_df
    h.sort_values("DATE").tail(4 * 260).to_csv(FII_FILE, index=False)

def build_fii_json():
    if not os.path.exists(FII_FILE):
        return
    h = pd.read_csv(FII_FILE)
    if not len(h):
        return
    latest_d = h["DATE"].max()
    latest = {}
    for who in ["FII", "DII", "Client", "Pro"]:
        r = h[(h["DATE"] == latest_d) & (h["WHO"] == who)]
        if len(r):
            L, S = int(r["LONG"].iloc[0]), int(r["SHORT"].iloc[0])
            latest[who.lower()] = {"long": L, "short": S, "net": L - S,
                                   "long_pct": round(100 * L / (L + S), 1) if L + S else None}
    fii = h[h["WHO"] == "FII"].sort_values("DATE").tail(130)
    series = [{"d": r["DATE"], "net": int(r["LONG"] - r["SHORT"]),
               "lp": round(100 * r["LONG"] / (r["LONG"] + r["SHORT"]), 1) if r["LONG"] + r["SHORT"] else None}
              for _, r in fii.iterrows()]
    with open(os.path.join(DATA_DIR, "fii.json"), "w") as fh:
        json.dump({"asof": latest_d, "latest": latest, "series": series}, fh, separators=(",", ":"))

# ---------------------------------------------------------------- news (RSS, fetched by Actions)
FEEDS = [
    ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Livemint", "https://www.livemint.com/rss/markets"),
]

def fetch_news():
    import xml.etree.ElementTree as ET
    import html as html_mod
    from email.utils import parsedate_to_datetime
    items = []
    mock_dir = os.environ.get("NEWS_MOCK_DIR")
    sources = ([("Mock", os.path.join(mock_dir, f)) for f in sorted(os.listdir(mock_dir))]
               if mock_dir else FEEDS)
    for src, loc in sources:
        try:
            raw = open(loc, "rb").read() if mock_dir else requests.get(loc, headers=UA, timeout=25).content
            root = ET.fromstring(raw)
            for it in root.iter("item"):
                t = (it.findtext("title") or "").strip()
                lk = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                if not t or not lk:
                    continue
                try:
                    ts = parsedate_to_datetime(pub).isoformat()
                except Exception:
                    ts = ""
                items.append({"title": html_mod.unescape(t)[:200], "link": lk,
                              "src": src, "ts": ts})
        except Exception as e:
            print(f"news feed skipped ({src}): {type(e).__name__}")
    seen, out = set(), []
    for it in sorted(items, key=lambda x: x["ts"], reverse=True):
        k = it["title"].lower()[:80]
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    with open(os.path.join(DATA_DIR, "news.json"), "w") as fh:
        json.dump({"updated_utc": datetime.utcnow().isoformat() + "Z",
                   "items": out[:40]}, fh, separators=(",", ":"))
    print(f"news: {len(out[:40])} items from {len(sources)} feeds")

# ---------------------------------------------------------------- scans + market
SCAN_DEFS = {
    # Trend & Momentum
    "above_200ema":   ("Trading above 200-day EMA", "Trend", lambda x: x.get("ema200") and x["close"] > x["ema200"]),
    "golden_cross":   ("Golden cross (50>200) today", "Trend", lambda x: x.get("golden_cross")),
    "st_flip_bull":   ("Supertrend flipped bullish today", "Trend", lambda x: x.get("st_flip") and x["st_dir"] == 1),
    "st_flip_bear":   ("Supertrend flipped bearish today", "Trend", lambda x: x.get("st_flip") and x["st_dir"] == -1),
    "rsi_oversold":   ("RSI below 30 (oversold)", "Momentum", lambda x: x.get("rsi") is not None and x["rsi"] < 30),
    "rsi_overbought": ("RSI above 70 (overbought)", "Momentum", lambda x: x.get("rsi") is not None and x["rsi"] > 70),
    "macd_bull":      ("MACD above signal line", "Momentum", lambda x: x.get("macd") is not None and x["macd"] > x["macd_sig"]),
    # Price & Volume
    "vol_spike":      ("Volume 2x+ its 20-day average", "Price & Volume", lambda x: x.get("vol_x") and x["vol_x"] >= 2),
    "deliv_accum":    ("Rising delivery % + rising price", "Price & Volume", lambda x: x.get("deliv") and x.get("deliv_avg20")
                                                            and x["deliv"] > x["deliv_avg20"] + 5 and x["chg_pct"] > 0),
    "wk1_hi":         ("1-week high breakout", "Price & Volume", lambda x: x.get("wk1_hi")),
    "wk1_lo":         ("1-week low breakdown", "Price & Volume", lambda x: x.get("wk1_lo")),
    "wk4_hi":         ("4-week high breakout", "Price & Volume", lambda x: x.get("wk4_hi")),
    "wk4_lo":         ("4-week low breakdown", "Price & Volume", lambda x: x.get("wk4_lo")),
    "nr7":            ("Narrow range day (NR7)", "Price & Volume", lambda x: x.get("nr7")),
    # 52-Week Levels
    "hi_52w":         ("At / within 1% of 52-week high", "52-Week", lambda x: x.get("from_hi_pct") is not None and x["from_hi_pct"] >= -1),
    "lo_52w":         ("At / within 1% of 52-week low", "52-Week", lambda x: x.get("from_lo_pct") is not None and x["from_lo_pct"] <= 1),
    "near5_hi":       ("Within 5% of 52-week high", "52-Week", lambda x: x.get("from_hi_pct") is not None and -5 <= x["from_hi_pct"] < -1),
    "near5_lo":       ("Within 5% of 52-week low", "52-Week", lambda x: x.get("from_lo_pct") is not None and 1 < x["from_lo_pct"] <= 5),
    # Pivot Levels (from previous session's pivots)
    "x_r1": ("Crossed above Resistance R1", "Pivot Levels", lambda x: x.get("x_r1")),
    "x_r2": ("Crossed above Resistance R2", "Pivot Levels", lambda x: x.get("x_r2")),
    "x_r3": ("Crossed above Resistance R3", "Pivot Levels", lambda x: x.get("x_r3")),
    "x_s1": ("Broke below Support S1", "Pivot Levels", lambda x: x.get("x_s1")),
    "x_s2": ("Broke below Support S2", "Pivot Levels", lambda x: x.get("x_s2")),
    "x_s3": ("Broke below Support S3", "Pivot Levels", lambda x: x.get("x_s3")),
    # Candlestick Patterns
    "bull_engulf":    ("Bullish Engulfing", "Candlestick", lambda x: "bull_engulf" in x.get("patterns", [])),
    "bear_engulf":    ("Bearish Engulfing", "Candlestick", lambda x: "bear_engulf" in x.get("patterns", [])),
    "hammer":         ("Hammer", "Candlestick", lambda x: "hammer" in x.get("patterns", [])),
    "shooting_star":  ("Shooting Star", "Candlestick", lambda x: "shooting_star" in x.get("patterns", [])),
    "piercing":       ("Piercing Line", "Candlestick", lambda x: "piercing" in x.get("patterns", [])),
    "dark_cloud":     ("Dark Cloud Cover", "Candlestick", lambda x: "dark_cloud" in x.get("patterns", [])),
    "three_soldiers": ("Three White Soldiers", "Candlestick", lambda x: "three_soldiers" in x.get("patterns", [])),
    "three_crows":    ("Three Black Crows", "Candlestick", lambda x: "three_crows" in x.get("patterns", [])),
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
        for k2, v2 in list(ind.items()):
            if isinstance(v2, float) and v2 != v2:      # NaN -> null
                ind[k2] = None
        tail = hist.tail(260)
        def _col(vals, nd=2, as_int=False):
            out2 = []
            for v3 in vals:
                if pd.isna(v3):
                    out2.append(None)
                else:
                    out2.append(int(v3) if as_int else round(float(v3), nd))
            return out2
        payload = {"symbol": sym, "asof": hist["DATE"].iloc[-1], "ind": ind,
                   "series": {"date": tail["DATE"].tolist(),
                              "o": _col(tail["OPEN_PRICE"]),
                              "h": _col(tail["HIGH_PRICE"]),
                              "l": _col(tail["LOW_PRICE"]),
                              "c": _col(tail["CLOSE_PRICE"]),
                              "v": _col(tail["TTL_TRD_QNTY"], as_int=True),
                              "dlv": _col(tail["DELIV_PER"], nd=1)}}
        with open(os.path.join(STOCK_DIR, f"{sym}.json"), "w") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        row = {"symbol": sym, "close": ind["close"], "chg": ind["chg_pct"],
               "rsi": ind.get("rsi"), "vol_x": ind.get("vol_x"),
               "deliv": ind.get("deliv"), "ret_1m": ind.get("ret_1m")}
        universe.append(row)
        for k, (_, _cat, fn) in SCAN_DEFS.items():
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
                   "scans": [{"id": k, "name": SCAN_DEFS[k][0], "cat": SCAN_DEFS[k][1],
                              "count": len(v), "rows": v[:100]}
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
            idx = fetch_indices(d)
            if idx is not None:
                append_indices(idx)
                append_valuation(idx)
            fo = fetch_fii(d)
            if fo is not None:
                append_fii(fo)
            print(f"fetched {d} rows={len(df)} new_history_rows={n}")
            break
        d -= timedelta(days=1)
    else:
        print("no trading day found in the last 7 days — nothing to do")
    # seed index/sector history on first runs (needs 2+ days for change calc)
    probe = os.path.join(SECT_HIST, "IT.csv")
    if not os.path.exists(probe) or len(pd.read_csv(probe)) < 2:
        for i in range(12, 0, -1):
            dd = date.today() - timedelta(days=i)
            if dd.weekday() >= 5:
                continue
            idx2 = fetch_indices(dd)
            if idx2 is not None:
                append_indices(idx2)
                append_valuation(idx2)
            if not os.environ.get("NSE_MOCK_DIR"):
                time.sleep(0.5)
    # seed FII participant history for the long-short chart
    if not os.path.exists(FII_FILE) or len(pd.read_csv(FII_FILE)) < 20:
        for i in range(130, 0, -1):
            dd = date.today() - timedelta(days=i)
            if dd.weekday() >= 5:
                continue
            fo2 = fetch_fii(dd)
            if fo2 is not None:
                append_fii(fo2)
            if not os.environ.get("NSE_MOCK_DIR"):
                time.sleep(0.4)
    build_indices_json()
    build_sectors_json()
    build_fii_json()
    build_valuation_json()
    build_names_json()
    build_n50_map()
    build_holidays_json()
    build_index_pages()
    fetch_news()
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
        idx = fetch_indices(d)
        if idx is not None:
            append_indices(idx)
            append_valuation(idx)
        fo = fetch_fii(d)
        if fo is not None:
            append_fii(fo)
        got += 1
        if got % 20 == 0:
            print(f"...{got} trading days ingested (latest {d})")
        if not os.environ.get("NSE_MOCK_DIR"):
            time.sleep(0.6)                 # be polite to NSE
    print(f"backfill complete: {got} trading days")
    build_indices_json()
    build_sectors_json()
    build_fii_json()
    build_valuation_json()
    build_names_json()
    build_n50_map()
    build_holidays_json()
    build_index_pages()
    fetch_news()
    n, asof = rebuild_outputs()
    print(f"outputs rebuilt: {n} stocks, as of {asof}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "daily"
    if mode == "backfill":
        run_backfill(int(sys.argv[2]) if len(sys.argv) > 2 else 320)
    elif mode == "news":
        fetch_news()
    else:
        run_daily()
