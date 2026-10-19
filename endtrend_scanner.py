import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

TG_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")

# Indicator settings (same as the Pine script)
SIG_LEN = 14
DI_LEN = 14
HL_RANGE = 20
HL_TREND = 35

# MACD settings
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
MACD_CROSS = "bullish"   # "bullish", "bearish" or "any"
MACD_MIN_BACK = 1        # MACD cross must be 1..3 candles before the End Trend candle
MACD_MAX_BACK = 3


def get_lbank_symbols():
    url = "https://lbkperp.lbank.com/cfd/openApi/v1/pub/instrument?productGroup=SwapU"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    syms = []
    for it in data:
        s = it.get("symbol", "")
        if s.endswith("USDT"):
            syms.append(s)
    return sorted(set(syms))


def drop_forming(o):
    now = pd.Timestamp.now(tz="UTC")
    return o[o.index + pd.Timedelta(hours=4) <= now]


def get_4h_yahoo(symbol):
    base = symbol[:-4]
    ticker = f"{base}-USD"
    df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    o = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    o = drop_forming(o)
    return o if len(o) > 100 else None


def get_4h_lbank_spot(symbol):
    base = symbol[:-4].lower()
    pair = f"{base}_usdt"
    start = int(time.time()) - 300 * 4 * 3600
    r = requests.get(
        "https://api.lbkex.com/v2/kline.do",
        params={"symbol": pair, "size": 300, "type": "hour4", "time": start},
        timeout=30,
    )
    rows = r.json().get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Vol"])
    df["ts"] = pd.to_datetime(df["ts"].astype(float), unit="s", utc=True)
    df = df.set_index("ts").astype(float).sort_index()
    df = df[["Open", "High", "Low", "Close"]]
    df = drop_forming(df)
    return df if len(df) > 100 else None


def get_4h(symbol):
    """Try Yahoo first; if the coin is missing there, fall back to LBank spot."""
    source = "yahoo"
    df = None
    try:
        df = get_4h_yahoo(symbol)
    except Exception as e:
        print(symbol, "yahoo error:", e)
    if df is None:
        source = "lbank-spot"
        try:
            df = get_4h_lbank_spot(symbol)
        except Exception as e:
            print(symbol, "lbank spot error:", e)
    return (df, source) if df is not None else (None, None)


def rma(values, length):
    """Pine ta.rma: seeded with SMA of first `length` valid values."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < length:
        return out
    start = valid[0] + length - 1
    out[start] = np.nanmean(v[valid[0]:start + 1])
    a = 1.0 / length
    for i in range(start + 1, len(v)):
        x = v[i] if not np.isnan(v[i]) else 0.0
        out[i] = a * x + (1 - a) * out[i - 1]
    return out


def end_trend_on_last_bar(df):
    h = df["High"].values
    l = df["Low"].values
    c = df["Close"].values
    n = len(df)

    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    up[1:] = h[1:] - h[:-1]
    dn[1:] = -(l[1:] - l[:-1])

    plus_dm = np.where(np.isnan(up), np.nan, np.where((up > dn) & (up > 0), up, 0.0))
    minus_dm = np.where(np.isnan(dn), np.nan, np.where((dn > up) & (dn > 0), dn, 0.0))

    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    tr_rma = rma(tr, DI_LEN)
    plus = pd.Series(100 * rma(plus_dm, DI_LEN) / tr_rma).ffill().values
    minus = pd.Series(100 * rma(minus_dm, DI_LEN) / tr_rma).ffill().values

    s = plus + minus
    ratio = np.abs(plus - minus) / np.where(s == 0, 1, s)
    sig = 100 * rma(ratio, SIG_LEN)

    condition = 0.0
    prev = dict(hl=False, diUp=False, diDn=False, plus=np.nan, minus=np.nan, sig=np.nan)
    long_x = short_x = False

    for i in range(n):
        if np.isnan(sig[i]) or np.isnan(plus[i]) or np.isnan(minus[i]):
            prev = dict(hl=False, diUp=False, diDn=False, plus=plus[i], minus=minus[i], sig=sig[i])
            continue

        hl = sig[i] <= HL_RANGE
        di_up = plus[i] >= minus[i]
        di_dn = minus[i] > plus[i]
        di_up_up = plus[i] >= HL_TREND
        di_dn_dn = minus[i] > HL_TREND

        if not np.isnan(prev["plus"]) and not np.isnan(prev["minus"]):
            cross_di = (plus[i] > minus[i] and prev["plus"] <= prev["minus"]) or \
                       (plus[i] < minus[i] and prev["plus"] >= prev["minus"])
        else:
            cross_di = False
        sig_up = (not np.isnan(prev["sig"])) and sig[i] > prev["sig"]

        entry_long = (not hl and di_up and sig_up and not prev["diUp"]) or \
                     (not hl and di_up and sig_up and sig[i] > HL_RANGE and prev["hl"])
        entry_short = (not hl and di_dn and sig_up and not prev["diDn"]) or \
                      (not hl and di_dn and sig_up and sig[i] > HL_RANGE and prev["hl"])
        entry_long_str = not hl and di_up and sig_up and di_up_up
        entry_short_str = not hl and di_dn and sig_up and di_dn_dn
        exit_long = (cross_di and prev["diUp"]) or (hl and not prev["hl"])
        exit_short = (cross_di and prev["diDn"]) or (hl and not prev["hl"])

        old = condition
        if old != 1 and entry_long_str:
            condition = 1.0
        elif old != -1 and entry_short_str:
            condition = -1.0
        elif old != 0.5 and entry_long:
            condition = 0.5
        elif old != -0.5 and entry_short:
            condition = -0.5
        elif old != 0 and exit_long:
            condition = 0.0
        elif old != 0 and exit_short:
            condition = 0.0

        long_x = old in (0.5, 1.0) and condition == 0
        short_x = old in (-0.5, -1.0) and condition == 0

        prev = dict(hl=hl, diUp=di_up, diDn=di_dn, plus=plus[i], minus=minus[i], sig=sig[i])

    # state of the LAST completed candle
    if long_x:
        return "long"
    if short_x:
        return "short"
    return None


def macd_cross_distance(df):
    """Return k (1..MACD_MAX_BACK) if a MACD cross happened k candles before
    the last completed candle, else None. The nearest one wins."""
    close = df["Close"]
    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    m, s = macd.values, sig.values
    n = len(df)
    for k in range(MACD_MIN_BACK, MACD_MAX_BACK + 1):
        j = n - 1 - k
        if j < 1:
            break
        bull = m[j] > s[j] and m[j - 1] <= s[j - 1]
        bear = m[j] < s[j] and m[j - 1] >= s[j - 1]
        if (MACD_CROSS == "bullish" and bull) or \
           (MACD_CROSS == "bearish" and bear) or \
           (MACD_CROSS == "any" and (bull or bear)):
            return k
    return None


def send_telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    for i in range(0, len(text), 3800):
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={"chat_id": TG_CHAT, "text": text[i:i + 3800]},
            timeout=30,
        )


def main():
    symbols = get_lbank_symbols()
    print(f"{len(symbols)} LBank futures symbols")
    hits, skipped = [], 0
    for sym in symbols:
        try:
            df, source = get_4h(sym)
            if df is None:
                skipped += 1
                continue
            res = end_trend_on_last_bar(df)
            if res:
                dist = macd_cross_distance(df)
                if dist is not None:
                    hits.append((sym, res, df.index[-1], source, dist))
        except Exception as e:
            skipped += 1
            print(sym, "error:", e)
        time.sleep(0.3)

    if hits:
        lines = ["End Trend (X) - 4h last closed candle", ""]
        for sym, res, t, source, dist in hits:
            kind = "closing LONG trend" if res == "long" else "closing SHORT trend"
            lines.append(f"{sym}  -  {kind}  |  MACD cross {dist} candle(s) before  ({t:%Y-%m-%d %H:%M} UTC, {source})")
        lines.append("")
        lines.append(f"Scanned: {len(symbols) - skipped} | Skipped (no data): {skipped}")
        send_telegram("\n".join(lines))
    else:
        send_telegram(f"No End Trend signal on last 4h candle. Scanned: {len(symbols) - skipped}, skipped: {skipped}")


if __name__ == "__main__":
    main()
