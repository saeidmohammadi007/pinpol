import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import requests
import ccxt

# --- الگوی مرجع ---
PATTERN_SYMBOL   = 'ETH-USD'
PATTERN_DATE     = '2026-07-02'
SHOW_N           = 10

# --- فیلتر End Trend ---
FILTER_END_TREND = True
LOOKBACK         = 3        # چند کندل بسته‌شدهٔ اخیر برای دیدن End Trend
ADX_SMOOTHING    = 14
DI_LENGTH        = 14
LEVEL_RANGE      = 20
LEVEL_TREND      = 35

# --- داده ---
TIMEFRAME        = '4h'
TF_HOURS         = 4
OHLCV_LIMIT      = 500
DEMO_DEBUG_SYM   = 'MRK'
MAX_ERR_PRINT    = 5        # چند خطای اول چاپ شود

err_count = 0

# ---------- LBank ----------
def init_lbank():
    ex = ccxt.lbank({
        'options': {'defaultType': 'future'},
        'enableRateLimit': True,
    })
    ex.load_markets()
    return ex

def build_symbol_map(ex):
    out = {}
    for sym, m in ex.markets.items():
        if not m.get('swap'):
            continue
        base = m.get('base')
        quote = m.get('quote')
        if not base:
            continue
        b = base.upper()
        if quote == 'USDT':
            out[b] = sym
        elif b not in out:
            out[b] = sym
    return out

def fetch_4h(ex, unified):
    global err_count
    since = ex.milliseconds() - OHLCV_LIMIT * TF_HOURS * 3600 * 1000
    raw = None
    last_exc = None
    # اول با since، اگر نشد بدون since
    for kwargs in ({'since': since, 'limit': OHLCV_LIMIT}, {'limit': OHLCV_LIMIT}):
        try:
            raw = ex.fetch_ohlcv(unified, TIMEFRAME, **kwargs)
            if raw:
                break
        except Exception as e:
            last_exc = e
    if not raw:
        if last_exc is not None and err_count < MAX_ERR_PRINT:
            err_count += 1
            print(f"\n⚠️ خطا برای {unified}: {type(last_exc).__name__}: {last_exc}")
        return None
    if len(raw) < 60:
        return None
    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.drop_duplicates('ts').sort_values('ts')
    df.set_index('ts', inplace=True)
    df.index = df.index.tz_localize(None)
    return df[['open','high','low','close']]

def drop_forming(df, tf_hours=TF_HOURS):
    if df is None or len(df) == 0:
        return df
    now = pd.Timestamp.utcnow().tz_localize(None)
    if df.index[-1] + pd.Timedelta(hours=tf_hours) > now:
        return df.iloc[:-1]
    return df

# ---------- reference ----------
def get_ref():
    import yfinance as yf
    df = yf.download(PATTERN_SYMBOL, start='2024-01-01', interval='1d',
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open','High','Low','Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open','high','low','close']
    return df

def vec(o,h,l,c):
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None
    return np.array([(o-l)/rng, 1.0, 0.0, (c-l)/rng])

# ---------- ADX ----------
def adx_di(df, n_di=DI_LENGTH, n_adx=ADX_SMOOTHING):
    H, L, C = df['high'], df['low'], df['close']
    up, dn = H.diff(), -L.diff()
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0.0), index=df.index)
    tr = pd.concat([H-L, (H-C.shift()).abs(), (L-C.shift()).abs()], axis=1).max(axis=1)
    def rma(s,n): return s.ewm(alpha=1/n, adjust=False).mean()
    trr = rma(tr, n_di)
    pdi = 100 * rma(pdm, n_di) / trr
    mdi = 100 * rma(mdm, n_di) / trr
    dx  = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx.fillna(0), n_adx), pdi, mdi

def state_machine(adx, pdi, mdi, hr=LEVEL_RANGE, ht=LEVEL_TREND):
    a, p, m = adx.values, pdi.values, mdi.values
    n = len(a)
    cond = np.zeros(n)
    ex   = np.zeros(n, dtype=int)
    for i in range(1, n):
        hp, hp_p = a[i] <= hr, a[i-1] <= hr
        up_, up_p = p[i] >= m[i], p[i-1] >= m[i-1]
        dn_, dn_p = m[i] >  p[i], m[i-1] >  p[i-1]
        cross  = up_ != up_p
        rising = a[i] > a[i-1]

        e_long   = (not hp) and up_ and rising and (not up_p)
        e_short  = (not hp) and dn_ and rising and (not dn_p)
        e_long_s = (not hp) and up_ and rising and (p[i] >= ht)
        e_shrt_s = (not hp) and dn_ and rising and (m[i] >  ht)

        x_long  = (cross and up_p) or (hp and not hp_p)
        x_short = (cross and dn_p) or (hp and not hp_p)

        prev = cond[i-1]
        x_l_now = (prev > 0) and x_long
        x_s_now = (prev < 0) and x_short

        if x_l_now:   ex[i] =  1
        elif x_s_now: ex[i] = -1

        if   prev !=  1   and e_long_s:   cond[i] =  1
        elif prev != -1   and e_shrt_s:   cond[i] = -1
        elif prev !=  0.5 and e_long:     cond[i] =  0.5
        elif prev != -0.5 and e_short:    cond[i] = -0.5
        elif x_l_now or x_s_now:          cond[i] =  0
        else:                             cond[i] = prev

    return (pd.Series(cond, index=adx.index),
            pd.Series(ex, index=adx.index))

def lbl(v):
    return {1.0:"Long قوی", 0.5:"Long ضعیف", 0.0:"خنثی",
            -0.5:"Short ضعیف", -1.0:"Short قوی"}.get(float(v),"?")

def tg(text):
    tok = os.environ.get('TELEGRAM_BOT_TOKEN')
    cid = os.environ.get('TELEGRAM_CHAT_ID')
    if not tok or not cid:
        print("❌ TELEGRAM env vars not set"); return
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={'chat_id':cid,'text':text,'parse_mode':'HTML'},
                          timeout=10)
        print("✅ ارسال شد" if r.status_code==200 else f"⚠️ {r.text}")
    except Exception as e:
        print(f"❌ {e}")

# ---------- main ----------
print(f"🔍 کندل مرجع {PATTERN_SYMBOL} @ {PATTERN_DATE}")
ref = get_ref()
if ref is None: exit("❌ no ref")
mask = ref.index.date == pd.to_datetime(PATTERN_DATE).date()
if not mask.any(): exit(f"❌ کندل {PATTERN_DATE} موجود نیست")
cdl = ref[mask].iloc[0]
o,h,l,c = cdl['open'], cdl['high'], cdl['low'], cdl['close']
pv = vec(o,h,l,c)
if pv is None: exit("❌ invalid candle")
print(f"📌 الگو: O={o:.4f} H={h:.4f} L={l:.4f} C={c:.4f}")

print("\n📊 اتصال به LBank ...")
ex = init_lbank()
smap = build_symbol_map(ex)
print(f"✅ نمادها: {len(smap)}")

results = []
skip_nt = 0
skip_nd = 0

for base in tqdm(sorted(smap.keys()), desc="اسکن 4h (LBank)"):
    try:
        df = fetch_4h(ex, smap[base])
        if df is None or len(df) < 60:
            skip_nd += 1; continue

        adx, pdi, mdi = adx_di(df)
        if adx.isna().all():
            skip_nd += 1; continue

        cond, exs = state_machine(adx, pdi, mdi)

        df_closed = drop_forming(df)
        if len(df_closed) < 60:
            skip_nd += 1; continue
        last = len(df_closed) - 1   # آخرین کندل بسته‌شده

        # پیدا کردن آخرین End Trend در LOOKBACK کندل اخیر
        si = last
        ex_v = int(exs.iloc[last])
        if FILTER_END_TREND:
            si = None
            for i in range(last, max(last - LOOKBACK, 0), -1):
                if int(exs.iloc[i]) != 0:
                    si = i
                    break
            if si is None:
                skip_nt += 1; continue
            ex_v = int(exs.iloc[si])

        row = df.iloc[si]
        v = vec(row['open'], row['high'], row['low'], row['close'])
        if v is None: continue
        d = float(np.linalg.norm(pv - v))

        results.append({
            'sym': base, 'dist': d,
            'ts': df.index[si].strftime('%Y-%m-%d %H:%M UTC'),
            'ago': last - si,
            'o':float(row['open']), 'h':float(row['high']),
            'l':float(row['low']),  'c':float(row['close']),
            'adx':   float(adx.iloc[si]),
            'adx_p': float(adx.iloc[si-1]) if si >= 1 else float('nan'),
            'pdi':   float(pdi.iloc[si]),
            'mdi':   float(mdi.iloc[si]),
            'st_b':  float(cond.iloc[si-1]) if si >= 1 else 0.0,
            'st_a':  float(cond.iloc[si]),
            'ex':    ex_v,
        })

        if base == DEMO_DEBUG_SYM:
            print(f"\n[DEBUG {base}] کندل‌های اخیر (LBank):")
            for i in range(max(0, si-4), si+1):
                mark = " ←سیگنال" if i == si else ""
                print(f"  {df.index[i]} | ADX={adx.iloc[i]:.3f} "
                      f"+DI={pdi.iloc[i]:.2f} -DI={mdi.iloc[i]:.2f} "
                      f"state={cond.iloc[i]:+.1f} exit={int(exs.iloc[i])}{mark}")

        time.sleep(0.3)
    except Exception as e:
        if err_count < MAX_ERR_PRINT:
            err_count += 1
            print(f"\n⚠️ خطا در پردازش {base}: {type(e).__name__}: {e}")
        continue

# ---------- تشخیص ----------
print(f"\nno_data={skip_nd} | no_end_trend={skip_nt} | "
      f"total={len(smap)} | results={len(results)}")

# ---------- گزارش ----------
if not results:
    msg = (f"❌ هیچ کندلی با End Trend در {LOOKBACK} کندل بسته‌شدهٔ اخیر یافت نشد.\n"
           f"📊 اسکن: {len(smap)} | بدون داده: {skip_nd} | بدون End Trend: {skip_nt}")
    print("\n"+msg); tg(msg); exit()

df_r = pd.DataFrame(results).sort_values('dist').head(SHOW_N)

lines = [f"🏆 <b>مشابه الگوی {PATTERN_SYMBOL} ({PATTERN_DATE})</b>\n",
         f"الگو: O={o:.6g} H={h:.6g} L={l:.6g} C={c:.6g}",
         f"🔎 <i>داده مستقیم از LBank | End Trend در {LOOKBACK} کندل بسته‌شدهٔ اخیر 4h (UTC)</i>\n"]

for _, r in df_r.iterrows():
    ex_lbl = {1:"🟢 خروج از Long", -1:"🔴 خروج از Short"}.get(int(r['ex']),"")
    adx_p_s = f"{r['adx_p']:.2f}" if pd.notna(r['adx_p']) else "—"
    warn = " ⚠️ مرزی" if 19.5 <= r['adx'] <= 20.5 else ""
    ago_s = "آخرین کندل" if int(r['ago']) == 0 else f"{int(r['ago'])} کندل قبل"
    lines.append(
        f"🔸 <b>{r['sym']}</b> (فاصله: {r['dist']:.4f})\n"
        f"   کندل: {r['ts']} ({ago_s})\n"
        f"   O={r['o']:.6g} H={r['h']:.6g} L={r['l']:.6g} C={r['c']:.6g}\n"
        f"   ADX={r['adx']:.3f}{warn} (قبلی: {adx_p_s}) | "
        f"+DI={r['pdi']:.2f} | -DI={r['mdi']:.2f}\n"
        f"   {lbl(r['st_b'])} → {lbl(r['st_a'])} | {ex_lbl}"
    )

lines.append(f"\n📊 اسکن: {len(smap)} | بدون داده: {skip_nd} | "
             f"بدون End Trend: {skip_nt} | نتایج: {len(df_r)}")

msg = "\n".join(lines)
print("\n"+msg); tg(msg)
print("\n✅ Done")
