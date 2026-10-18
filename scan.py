import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import requests
import ccxt

# --- الگوی مرجع: تک‌کندل روزانه ETH ---
PATTERN_SYMBOL   = 'ETH-USD'
PATTERN_DATE     = '2026-07-02'
SHOW_N           = 10

# --- فیلتر End Trend ---
FILTER_END_TREND = True
ADX_SMOOTHING    = 14
DI_LENGTH        = 14
LEVEL_RANGE      = 20
LEVEL_TREND      = 35

# ---------- توابع ----------
def get_lbank_futures_symbols():
    exchange = ccxt.lbank({'options': {'defaultType': 'future'}})
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"❌ خطا در اتصال به LBank: {e}")
        return []

    base_list = []
    for symbol, market in markets.items():
        if not market.get('swap'):
            continue
        base = market.get('base')
        if not base or base.isdigit():
            continue
        base_list.append(base.upper())

    seen, unique_bases = set(), []
    for b in base_list:
        if b not in seen:
            seen.add(b)
            unique_bases.append(b)
    print(f"✅ تعداد ارزهای پایه‌ی منحصربه‌فرد فیوچرز LBank: {len(unique_bases)}")
    return unique_bases

def get_daily_data(ticker, start='2024-01-01'):
    df = yf.download(ticker, start=start, interval='1d',
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']
    return df

def get_4h_data(ticker):
    df = yf.download(ticker, period='60d', interval='1h',
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df_4h = df.resample('4h').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
    }).dropna()
    return df_4h

def candle_vector(o, h, l, c):
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None
    return np.array([
        (float(o) - float(l)) / rng,
        1.0,
        0.0,
        (float(c) - float(l)) / rng,
    ])

def compute_adx_di(df, di_len=DI_LENGTH, adx_len=ADX_SMOOTHING):
    high  = df['high']
    low   = df['low']
    close = df['close']

    up   = high.diff()
    down = -low.diff()

    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def rma(s, n):
        return s.ewm(alpha=1.0 / n, adjust=False).mean()

    tr_rma   = rma(tr, di_len)
    plus_di  = 100 * rma(plus_dm,  di_len) / tr_rma
    minus_di = 100 * rma(minus_dm, di_len) / tr_rma

    denom = (plus_di + minus_di).replace(0, np.nan)
    dx    = 100 * (plus_di - minus_di).abs() / denom
    adx   = rma(dx.fillna(0), adx_len)

    return adx, plus_di, minus_di

def compute_condition_state(adx, di_plus, di_minus,
                            hl_range=LEVEL_RANGE, hl_trend=LEVEL_TREND):
    """
    ماشین حالت + خروجی exit_sig برای هر کندل.

    ✅ اصلاح‌شده:
    - exit_sig فقط زمانی ثبت می‌شود که واقعاً در پوزیشن هم‌جهت باشیم.
    - شرط عمومی (hp و not hp_prev) بر اساس جهت پوزیشن تفکیک می‌شود
      تا هم‌زمان هم exit_long و هم exit_short فعال نشوند.

    exit_sig[i] =  1  → خروج از Long در کندل i
    exit_sig[i] = -1  → خروج از Short در کندل i
    exit_sig[i] =  0  → خروجی رخ نداده یا پوزیشن فعالی نبوده
    """
    adx_a = adx.values
    dp_a  = di_plus.values
    dm_a  = di_minus.values
    n     = len(adx_a)
    cond     = np.zeros(n)
    exit_sig = np.zeros(n, dtype=int)

    for i in range(1, n):
        hp      = adx_a[i]   <= hl_range
        hp_prev = adx_a[i-1] <= hl_range
        dup     = dp_a[i]    >= dm_a[i]
        dup_p   = dp_a[i-1]  >= dm_a[i-1]
        ddn     = dm_a[i]    >  dp_a[i]
        ddn_p   = dm_a[i-1]  >  dp_a[i-1]
        cross   = dup != dup_p
        sig_up  = adx_a[i] > adx_a[i-1]

        entry_long     = (not hp) and dup     and sig_up and (not dup_p)
        entry_short    = (not hp) and ddn     and sig_up and (not ddn_p)
        entry_long_str = (not hp) and dup     and sig_up and (dp_a[i] >= hl_trend)
        entry_short_st = (not hp) and ddn     and sig_up and (dm_a[i] >  hl_trend)

        # شرط‌های عمومی خروج (جهت‌دار نیستند)
        exit_long_cond  = (cross and dup_p) or (hp and not hp_prev)
        exit_short_cond = (cross and ddn_p) or (hp and not hp_prev)

        prev = cond[i-1]

        # ✅ فقط اگر واقعاً در پوزیشن هم‌جهت باشیم، خروج معتبر است
        exit_long_now  = (prev > 0) and exit_long_cond
        exit_short_now = (prev < 0) and exit_short_cond

        if exit_long_now:
            exit_sig[i] = 1
        elif exit_short_now:
            exit_sig[i] = -1

        # --- ماشین حالت ---
        if   prev !=  1   and entry_long_str:   cond[i] =  1
        elif prev != -1   and entry_short_st:   cond[i] = -1
        elif prev !=  0.5 and entry_long:       cond[i] =  0.5
        elif prev != -0.5 and entry_short:      cond[i] = -0.5
        elif exit_long_now:                     cond[i] =  0
        elif exit_short_now:                    cond[i] =  0
        else:                                   cond[i] = prev

    return pd.Series(cond, index=adx.index), pd.Series(exit_sig, index=adx.index)

def state_label(v):
    return {
         1.0: "Long قوی",
         0.5: "Long ضعیف",
         0.0: "خنثی",
        -0.5: "Short ضعیف",
        -1.0: "Short قوی",
    }.get(float(v), "?")

def send_telegram_message(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ توکن یا chat_id تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ خطا در ارسال پیام: {r.text}")
        else:
            print("✅ پیام با موفقیت به تلگرام ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")

# ---------- اجرای اصلی ----------
print(f"🔍 استخراج کندل مرجع {PATTERN_SYMBOL} در تاریخ {PATTERN_DATE} (روزانه) ...")
ref_daily = get_daily_data(PATTERN_SYMBOL, start='2024-01-01')
if ref_daily is None:
    print(f"❌ خطا در دریافت داده‌های {PATTERN_SYMBOL}")
    exit()

target_date = pd.to_datetime(PATTERN_DATE).date()
mask = ref_daily.index.date == target_date
if not mask.any():
    print(f"❌ کندلی برای تاریخ {PATTERN_DATE} یافت نشد.")
    print(f"📅 آخرین تاریخ موجود در داده: {ref_daily.index[-1].date()}")
    print(f"📅 اولین تاریخ موجود در داده: {ref_daily.index[0].date()}")
    exit()

candle = ref_daily[mask].iloc[0]
o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
pattern_vec = candle_vector(o, h, l, c)
if pattern_vec is None:
    print("❌ کندل مرجع نامعتبر است.")
    exit()

print(f"📌 کندل مرجع (روزانه): O={o:.4f}  H={h:.4f}  L={l:.4f}  C={c:.4f}")
print(f"📐 بردار الگو: O={pattern_vec[0]:.3f} | H={pattern_vec[1]:.3f} | "
      f"L={pattern_vec[2]:.3f} | C={pattern_vec[3]:.3f}")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
symbols = get_lbank_futures_symbols()
if not symbols:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

results = []
skipped_no_end_trend = 0

for sym in tqdm(symbols, desc="اسکن ۴ ساعته + End Trend (همان کندل -۲)"):
    try:
        df_4h = get_4h_data(f"{sym}-USD")
        if df_4h is None or len(df_4h) < 40:
            continue

        # --- محاسبه ADX / DI / condition + exit_sig ---
        adx, di_plus, di_minus = compute_adx_di(df_4h)
        if adx.isna().all() or len(adx) < 40:
            continue
        cond, exit_sig = compute_condition_state(adx, di_plus, di_minus)

        # ============================================================
        # ✅ End Trend واقعی روی همان کندل -۲
        #    exit_sig فقط وقتی != 0 می‌شود که پوزیشن فعالی بسته شود
        # ============================================================
        exit_now  = int(exit_sig.iloc[-2])   # 1=خروج از Long, -1=خروج از Short, 0=ندارد
        end_trend = (exit_now != 0)

        if FILTER_END_TREND and not end_trend:
            skipped_no_end_trend += 1
            continue

        # --- شباهت کندلی روی همان کندل -۲ ---
        prev = df_4h.iloc[-2]
        vec = candle_vector(prev['open'], prev['high'],
                            prev['low'],  prev['close'])
        if vec is None:
            continue

        dist = float(np.linalg.norm(pattern_vec - vec))

        # وضعیت قبل از خروج (کندل -۳) برای نمایش شفاف
        state_before = float(cond.iloc[-3]) if len(cond) >= 3 else 0.0
        state_after  = float(cond.iloc[-2])

        results.append({
            'symbol':     sym,
            'dist':       dist,
            'last_4h':    df_4h.index[-2].strftime('%Y-%m-%d %H:%M'),
            'o': float(prev['open']),
            'h': float(prev['high']),
            'l': float(prev['low']),
            'c': float(prev['close']),
            'adx':     float(adx.iloc[-2]),
            'di_plus': float(di_plus.iloc[-2]),
            'di_minus':float(di_minus.iloc[-2]),
            'state_before': state_before,
            'state_after':  state_after,
            'exit_type': exit_now,
            'end_trend': end_trend,
        })
        time.sleep(0.3)
    except Exception:
        continue

if results:
    df_res = pd.DataFrame(results).sort_values('dist').head(SHOW_N)

    lines = []
    lines.append(f"🏆 <b>کندل‌های ۴ ساعته (قبلی) مشابه کندل روزانه {PATTERN_SYMBOL} ({PATTERN_DATE})</b>\n")
    lines.append(f"الگو (روزانه): O={o:.6g} | H={h:.6g} | L={l:.6g} | C={c:.6g}")
    if FILTER_END_TREND:
        lines.append("🔎 <i>فقط کندل‌هایی که End Trend واقعی در همان کندل -۲ رخ داده</i>\n")
    else:
        lines.append("🔎 <i>بدون فیلتر End Trend</i>\n")

    for _, row in df_res.iterrows():
        exit_label = {
             1: "🟢 End Trend: خروج از Long",
            -1: "🔴 End Trend: خروج از Short",
             0: "—"
        }.get(int(row['exit_type']), "—")

        lines.append(
            f"🔸 <b>{row['symbol']}</b>  (فاصله: {row['dist']:.4f})\n"
            f"   زمان ۴h: {row['last_4h']} | "
            f"O={row['o']:.6g} H={row['h']:.6g} "
            f"L={row['l']:.6g} C={row['c']:.6g}\n"
            f"   ADX={row['adx']:.2f} | +DI={row['di_plus']:.2f} | "
            f"−DI={row['di_minus']:.2f}\n"
            f"   وضعیت قبلی: {state_label(row['state_before'])} → "
            f"وضعیت بعدی: {state_label(row['state_after'])}\n"
            f"   {exit_label}"
        )

    lines.append(f"\n📅 تعداد ارزهای اسکن‌شده: {len(symbols)}")
    if FILTER_END_TREND:
        lines.append(f"🚫 رد شده به دلیل نداشتن End Trend واقعی: {skipped_no_end_trend}")
    lines.append(f"✅ نتایج یافت‌شده: {len(df_res)}")

    message = "\n".join(lines)
    send_telegram_message(message)
    print("\n" + message)
else:
    print("\n❌ نتیجه‌ای یافت نشد.")
    send_telegram_message(
        "❌ در اسکن امروز هیچ کندلی با End Trend واقعی در کندل -۲ پیدا نشد."
        if FILTER_END_TREND else
        "❌ در اسکن امروز هیچ نتیجه‌ای یافت نشد."
    )

print("\n✅ اسکن کامل شد!")
