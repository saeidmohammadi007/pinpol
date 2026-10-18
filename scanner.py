"""
LBank futures scanner (4h)
شرط: روی آخرین کندل کامل‌شده، MACD سیگنال را رو به بالا کراس کرده
      و در LOOKBACK کندل اخیر (بدون قید قبل/بعد کراس)، مثلث کوچک قرمز
      اندیکاتور ADX Di+ Di- [Gu5] (shortE) داشته.
نتیجه به تلگرام فرستاده می‌شود (با تاریخ و ساعت کراس و مثلث).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ───────── تنظیمات (با env قابل تغییر) ─────────
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
LOOKBACK      = int(os.getenv("LOOKBACK", "20"))       # مثلث قرمز: در چند کندل اخیر
MACD_LOOKBACK = int(os.getenv("MACD_LOOKBACK", "3"))   # کراس مکدی: ۱ تا ۳ کندل قبل
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9
ADX_SIG_LEN, DI_LEN, LVL_RANGE, LVL_TREND = 14, 14, 20, 35
TF_SECONDS = 4 * 3600
NOTIFY_EMPTY = os.getenv("NOTIFY_EMPTY", "1") == "1"

PERP  = "https://lbkperp.lbank.com"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
RANGE_DAYS = int(os.getenv("RANGE_DAYS", "60"))
S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"


def get_json(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1 + i)
    return None


# ───────── لیست ارزهای فیوچرز LBank ─────────
def yahoo_ticker(base: str) -> str:
    b = base.upper()
    for pre in ("10000000", "1000000", "100000", "10000", "1000", "1M"):
        if b.startswith(pre) and len(b) > len(pre):
            b = b[len(pre):]
            break
    return f"{b}-USD"


def futures_symbols():
    j = get_json(f"{PERP}/cfd/openApi/v1/pub/instrument", {"productGroup": "SwapU"})
    if not j:
        sys.exit("cannot fetch LBank futures instruments")
    data = j.get("data", j) if isinstance(j, dict) else j
    out = []
    for it in data:
        base = (it.get("baseCurrency") or "").strip()
        if base:
            out.append((it["symbol"], yahoo_ticker(base)))
    return out


# ───────── کندل‌ها از یاهو (۱ ساعته -> ۴ ساعته) ─────────
def to_4h(df: pd.DataFrame, now=None) -> pd.DataFrame:
    now = now or time.time()
    d = df.copy()
    d["dt"] = pd.to_datetime(d["t"], unit="s", utc=True)
    d = d.set_index("dt")
    g = d.resample("4h", origin="epoch").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "t": "first"}
    )
    g["n"] = d["c"].resample("4h", origin="epoch").count()
    g = g[g["n"] == 4].dropna()
    g["t"] = ((g.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(float)
    g = g[g["t"] + TF_SECONDS <= now]
    return g[["t", "o", "h", "l", "c"]].reset_index(drop=True)


def fetch_klines(ticker):
    j = get_json(f"{YAHOO}/{ticker}", {"interval": "1h", "range": f"{RANGE_DAYS}d"})
    try:
        r = j["chart"]["result"][0]
        q = r["indicators"]["quote"][0]
        df = pd.DataFrame({"t": r["timestamp"], "o": q["open"], "h": q["high"],
                           "l": q["low"], "c": q["close"]}).dropna()
        df = df.astype(float).drop_duplicates("t").sort_values("t")
    except Exception:
        return None
    if df.empty:
        return None
    return to_4h(df)


# ───────── اندیکاتورها ─────────
def rma(x: pd.Series, n: int) -> pd.Series:
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    valid = np.where(~np.isnan(v))[0]
    if len(valid) < n:
        return pd.Series(out, index=x.index)
    s = valid[0] + n - 1
    out[s] = np.nanmean(v[valid[0]:s + 1])
    for i in range(s + 1, len(v)):
        out[i] = (out[i - 1] * (n - 1) + v[i]) / n if not np.isnan(v[i]) else np.nan
    return pd.Series(out, index=x.index)


def adx_short_triangle(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["h"], df["l"], df["c"]
    up = h.diff()
    down = -l.diff()
    plus_dm  = np.where(up.isna(),   np.nan, np.where((up > down) & (up > 0), up, 0.0))
    minus_dm = np.where(down.isna(), np.nan, np.where((down > up) & (down > 0), down, 0.0))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1, skipna=False)
    tr_r = rma(tr, DI_LEN)
    plus  = (100 * rma(pd.Series(plus_dm,  index=df.index), DI_LEN) / tr_r).ffill()
    minus = (100 * rma(pd.Series(minus_dm, index=df.index), DI_LEN) / tr_r).ffill()
    sm = plus + minus
    sig = 100 * rma((plus - minus).abs() / sm.where(sm != 0, 1), ADX_SIG_LEN)

    n = len(df)
    P, M, G = plus.to_numpy(), minus.to_numpy(), sig.to_numpy()
    short_e = np.zeros(n, dtype=bool)
    cond = 0.0
    for i in range(1, n):
        if np.isnan(G[i]) or np.isnan(G[i - 1]) or np.isnan(P[i]) or np.isnan(M[i]):
            continue
        hl       = G[i]     <= LVL_RANGE
        hl_prev  = G[i - 1] <= LVL_RANGE
        di_up    = P[i] >= M[i]
        di_dn    = M[i] >  P[i]
        di_dn_prev = (M[i - 1] >  P[i - 1]) if not np.isnan(M[i - 1]) else False
        di_up_prev = (P[i - 1] >= M[i - 1]) if not np.isnan(M[i - 1]) else False
        sig_up = G[i] > G[i - 1]
        cross = (P[i] > M[i] and P[i - 1] <= M[i - 1]) or (P[i] < M[i] and P[i - 1] >= M[i - 1])

        entry_long  = (not hl and di_up and sig_up and not di_up_prev) or \
                      (not hl and di_up and sig_up and G[i] > LVL_RANGE and hl_prev)
        entry_short = (not hl and di_dn and sig_up and not di_dn_prev) or \
                      (not hl and di_dn and sig_up and G[i] > LVL_RANGE and hl_prev)
        entry_long_str  = not hl and di_up and sig_up and P[i] >= LVL_TREND
        entry_short_st  = not hl and di_dn and sig_up and M[i] >  LVL_TREND
        exit_long  = (cross and di_up_prev) or (hl and not hl_prev)
        exit_short = (cross and di_dn_prev) or (hl and not hl_prev)

        prev = cond
        if   prev != 1    and entry_long_str:  cond = 1
        elif prev != -1   and entry_short_st:  cond = -1
        elif prev != 0.5  and entry_long:      cond = 0.5
        elif prev != -0.5 and entry_short:     cond = -0.5
        elif prev != 0    and exit_long:       cond = 0
        elif prev != 0    and exit_short:      cond = 0
        short_e[i] = (prev != -0.5 and cond == -0.5)
    return pd.Series(short_e, index=df.index)


def macd_cross_up(df: pd.DataFrame) -> pd.Series:
    c = df["c"]
    macd = c.ewm(span=MACD_FAST, adjust=False).mean() - c.ewm(span=MACD_SLOW, adjust=False).mean()
    sgn  = macd.ewm(span=MACD_SIG,  adjust=False).mean()
    return (macd.shift(1) <= sgn.shift(1)) & (macd > sgn)


def check(df: pd.DataFrame, stats=None):
    """df فقط شامل کندل‌های کامل‌شده است؛ آخرین ردیف = آخرین کندل کامل (شماره ۱).
    شرط ۱: مکدی در MACD_LOOKBACK کندل اخیر سیگنال را رو به بالا کراس کرده باشد.
    شرط ۲: مثلث قرمز ADX در LOOKBACK کندل اخیر وجود داشته باشد
          (بدون قید قبل/هم‌زمان/بعد از کراس).
    خروجی شامل تایم‌استمپ (UTC) هر کراس و هر مثلث است.
    """
    if df is None or len(df) < 100:
        return None
    stats = stats if stats is not None else {}
    stats["checked"] = stats.get("checked", 0) + 1

    cross = macd_cross_up(df).to_numpy()
    tri   = adx_short_triangle(df).to_numpy()
    n     = len(df)

    # کراس در ۱..MACD_LOOKBACK کندل اخیر
    cross_pos = [n - k for k in range(1, MACD_LOOKBACK + 1) if cross[n - k]]
    if not cross_pos:
        return None
    stats["macd_cross"] = stats.get("macd_cross", 0) + 1

    # شرط ۲: مثلث در LOOKBACK کندل اخیر
    tri_pos = [n - d for d in range(1, LOOKBACK + 1) if n - d >= 0 and tri[n - d]]
    if not tri_pos:
        return None
    stats["cross_with_tri"] = stats.get("cross_with_tri", 0) + 1

    # دیاگنوستیک: چند تا از این‌ها شرط قدیمی (مثلث فقط قبل از کراس) را داشتند؟
    old_ok = False
    for j in cross_pos:
        if any(j - d >= 0 and tri[j - d] for d in range(1, LOOKBACK + 1)):
            old_ok = True
            break
    if old_ok:
        stats["cross_with_tri_before"] = stats.get("cross_with_tri_before", 0) + 1

    return {
        "close":        df["c"].iloc[-1],
        "macd_ago":     [n - j for j in cross_pos],
        "macd_ts":      [int(df["t"].iloc[j]) for j in cross_pos],
        "tri_bars_ago": [n - j for j in tri_pos],
        "tri_ts":       [int(df["t"].iloc[j]) for j in tri_pos],
        "t":            pd.to_datetime(df["t"].iloc[-1], unit="s"),
    }


def telegram(text):
    if not TG_TOKEN or not TG_CHAT:
        print(text)
        return
    for i in range(0, len(text), 3800):
        try:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                          data={"chat_id": TG_CHAT, "text": text[i:i + 3800]}, timeout=20)
        except Exception as e:
            print(f"telegram send error: {e}")


def fmt_ts(ts):
    """تایم‌استمپ UTC -> رشته تاریخ/ساعت (شروع کندل)."""
    return pd.to_datetime(ts, unit="s").strftime("%Y-%m-%d %H:%M")


def main():
    print(f"config: LOOKBACK={LOOKBACK} MACD_LOOKBACK={MACD_LOOKBACK} RANGE_DAYS={RANGE_DAYS}")
    syms = futures_symbols()
    print(f"{len(syms)} futures symbols")
    hits, skipped, stats, last_t = [], 0, {}, None
    for sym, yt in syms:
        df = fetch_klines(yt)
        if df is None:
            skipped += 1
            continue
        if df is not None and len(df):
            last_t = max(last_t or 0, df["t"].iloc[-1])
        r = check(df, stats)
        if r:
            hits.append((sym, r))
            print("HIT", sym, r)
        time.sleep(0.3)

    lt = f"{pd.to_datetime(last_t, unit='s'):%Y-%m-%d %H:%M} UTC" if last_t else "?"
    summary = (f"آخرین کندل کامل: {lt}\n"
               f"ارز فیوچرز: {len(syms)} | بدون داده/تیکر یاهو: {len(syms) - stats.get('checked', 0)}\n"
               f"بررسی‌شده: {stats.get('checked', 0)}\n"
               f"MACD کراس صعودی (در {MACD_LOOKBACK} کندل اخیر): {stats.get('macd_cross', 0)}\n"
               f"  ├─ دارای مثلث در {LOOKBACK} کندل اخیر (شرط جدید): {stats.get('cross_with_tri', 0)}\n"
               f"  └─ [دیاگنوستیک] دارای مثلث فقط قبل از کراس (شرط قدیمی): {stats.get('cross_with_tri_before', 0)}")
    print(summary)

    if hits:
        lines = [f"📈 MACD کراس صعودی + مثلث قرمز ADX (۴ساعته)\n"
                 f"آخرین کندل کامل (شماره ۱): {hits[0][1]['t']:%Y-%m-%d %H:%M} UTC\n"]
        for sym, r in hits:
            macd_times = " , ".join(f"{fmt_ts(t)} UTC" for t in r["macd_ts"])
            tri_times  = " , ".join(f"{fmt_ts(t)} UTC" for t in r["tri_ts"])
            lines.append(
                f"• {sym}  قیمت: {r['close']:g}\n"
                f"   🕐 کراس مکدی: {macd_times}\n"
                f"   🔺 مثلث قرمز: {tri_times}"
            )
        lines.append("\n" + summary)
        telegram("\n".join(lines))
    elif NOTIFY_EMPTY:
        telegram("موردی پیدا نشد.\n" + summary)
    print(f"done. hits={len(hits)} skipped={skipped}")


if __name__ == "__main__":
    main()
