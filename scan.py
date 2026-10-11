import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import requests
import ccxt

# ---------- تنظیمات A+ ----------
TIMEFRAME       = '4h'
EMA_FAST        = 50
EMA_SLOW        = 200
WICK_RATIO      = 3.0
BODY_MAX_RATIO  = 0.25
CLOSE_ZONE      = 0.25
PULLBACK_TOL    = 0.005
MIN_BARS        = 220
EMA_SLOPE_LOOKBACK = 3
MIN_RR          = 2.0

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


def get_data(ticker):
    try:
        df = yf.download(ticker, period='60d', interval='1h',
                         progress=False, auto_adjust=False)
    except Exception:
        return None

    if df is None or df.empty:
        return None

    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df_res = df.resample('4h').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
    }).dropna()

    return df_res


def add_emas(df, fast=EMA_FAST, slow=EMA_SLOW):
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    return df


def is_bull_pin_bar(row):
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    rng = h - l
    if rng <= 0:
        return False

    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if body / rng > BODY_MAX_RATIO:
        return False
    if lower_wick < WICK_RATIO * body:
        return False
    if lower_wick < WICK_RATIO * upper_wick:
        return False
    if (c - l) / rng < (1 - CLOSE_ZONE):
        return False

    return True


def near_ema50(row, tol=PULLBACK_TOL):
    ema = row['ema_fast']
    if ema <= 0 or not np.isfinite(ema):
        return False
    return abs(row['low'] - ema) / ema <= tol


def is_uptrend(row):
    if not (np.isfinite(row['ema_fast']) and np.isfinite(row['ema_slow'])):
        return False
    if row['ema_fast'] <= row['ema_slow']:
        return False
    if row['close'] <= row['ema_slow']:
        return False
    if (row['ema_fast'] - row['ema_slow']) / row['ema_slow'] < 0.005:
        return False
    return True


def ema50_rising(df, lookback=EMA_SLOPE_LOOKBACK):
    if len(df) < lookback + 2:
        return False
    now_val  = df['ema_fast'].iloc[-2]
    prev_val = df['ema_fast'].iloc[-2 - lookback]
    if not (np.isfinite(now_val) and np.isfinite(prev_val)):
        return False
    return now_val > prev_val


def has_pullback_before(df, lookback=5):
    if len(df) < lookback + 3:
        return False
    before = df.iloc[-4:-2]
    down_count = sum(1 for _, r in before.iterrows() if r['close'] < r['open'])
    return down_count >= 1


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
now_utc = pd.Timestamp.utcnow().tz_localize(None)
print(f"🕒 زمان فعلی UTC: {now_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"⭐ اسکن ستاپ‌های A+ (پین بار صعودی سخت‌گیرانه در پولبک EMA{EMA_FAST})")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
symbols = get_lbank_futures_symbols()
if not symbols:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

signals = []
rejected_stats = {
    'trend': 0,
    'slope': 0,
    'pullback': 0,
    'pinbar': 0,
    'prev_pullback': 0,
    'rr': 0,
}

for sym in tqdm(symbols, desc="اسکن A+"):
    try:
        df = get_data(f"{sym}-USD")
        if df is None or len(df) < MIN_BARS:
            continue

        df = add_emas(df)
        row = df.iloc[-2]
        row_time = df.index[-2]

        if not is_uptrend(row):
            rejected_stats['trend'] += 1
            continue

        if not ema50_rising(df):
            rejected_stats['slope'] += 1
            continue

        if not near_ema50(row):
            rejected_stats['pullback'] += 1
            continue

        if not is_bull_pin_bar(row):
            rejected_stats['pinbar'] += 1
            continue

        if not has_pullback_before(df):
            rejected_stats['prev_pullback'] += 1
            continue

        entry = row['high']
        stop  = row['low']
        risk  = entry - stop
        if risk <= 0:
            continue

        tp1 = entry + risk * 1.0
        tp2 = entry + risk * 2.0

        rr = (tp2 - entry) / risk
        if rr < MIN_RR:
            rejected_stats['rr'] += 1
            continue

        signals.append({
            'symbol': sym,
            'time': row_time.strftime('%Y-%m-%d %H:%M'),
            'o': float(row['open']),
            'h': float(row['high']),
            'l': float(row['low']),
            'c': float(row['close']),
            'ema50': float(row['ema_fast']),
            'ema200': float(row['ema_slow']),
            'entry': float(entry),
            'stop': float(stop),
            'tp1': float(tp1),
            'tp2': float(tp2),
        })
        time.sleep(0.2)
    except Exception:
        continue

# ---------- ارسال به تلگرام ----------
if signals:
    lines = []
    lines.append(f"⭐ <b>ستاپ‌های A+ — پین بار صعودی در پولبک EMA{EMA_FAST}</b>")
    lines.append(f"⏱ تایم‌فریم: {TIMEFRAME} | "
                 f"آخرین کندل کامل‌شده: {signals[0]['time']} UTC")
    lines.append(f"⚙️ سایه≥{WICK_RATIO}x | بدنه≤{BODY_MAX_RATIO} | "
                 f"پولبک≤{PULLBACK_TOL*100}%\n")

    for s in signals:
        lines.append(
            f"━━━━━━━━━━━━━━━\n"
            f"🌟 <b>{s['symbol']}</b> — 🟢 خرید A+\n"
            f"🕒 کندل: {s['time']}\n"
            f"📊 OHLC: O={s['o']:.6g} H={s['h']:.6g} "
            f"L={s['l']:.6g} C={s['c']:.6g}\n"
            f"📈 EMA50={s['ema50']:.6g} | EMA200={s['ema200']:.6g}\n"
            f"🎯 ورود: {s['entry']:.6g}\n"
            f"🛑 استاپ: {s['stop']:.6g}\n"
            f"✅ TP1 (1R): {s['tp1']:.6g}\n"
            f"✅ TP2 (2R): {s['tp2']:.6g}"
        )

    lines.append(f"\n📅 تعداد ارزهای اسکن‌شده: {len(symbols)}")
    lines.append(f"⭐ تعداد ستاپ A+: {len(signals)}")
    lines.append(f"🕒 زمان اجرا: {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    message = "\n".join(lines)
    send_telegram_message(message)
    print("\n" + message)
else:
    msg = (f"⭐ اسکن A+ (پین بار صعودی EMA{EMA_FAST}) انجام شد.\n"
           f"❌ هیچ ستاپ A+ یافت نشد.\n"
           f"📊 آمار رد شدن:\n"
           f"  • روند نامناسب: {rejected_stats['trend']}\n"
           f"  • شیب EMA50 صعودی نیست: {rejected_stats['slope']}\n"
           f"  • پولبک دقیق به EMA50: {rejected_stats['pullback']}\n"
           f"  • پین بار ضعیف: {rejected_stats['pinbar']}\n"
           f"  • اصلاح قبل پین بار: {rejected_stats['prev_pullback']}\n"
           f"📅 تعداد ارزهای اسکن‌شده: {len(symbols)}\n"
           f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    send_telegram_message(msg)
    print("\n" + msg)

print("\n✅ اسکن کامل شد!")
