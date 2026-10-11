import os
import pandas as pd
import numpy as np
import ccxt
import time
import requests
from tqdm import tqdm

# ---------- تنظیمات ----------
TIMEFRAMES      = ['1d']     # ← فقط روزانه
MIN_BARS        = 30         # ← چون فقط روزانه لازم داریم (1000 ساعت ≈ 41 روز)
MIN_SIMILARITY  = 75.0       # ← حداقل درصد شباهت به پین‌بار ایده‌آل

IDEAL_VECTOR    = np.array([0.0, 0.0, 1.0])
MAX_DIST        = np.sqrt(2.0)

# ---------- اتصال ccxt ----------
_EXCHANGE = None

def get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = ccxt.lbank({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })
    return _EXCHANGE


def get_lbank_futures_symbols():
    exchange = get_exchange()
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"❌ خطا در اتصال به LBank: {e}")
        return []

    symbols = []
    for symbol, market in markets.items():
        if not market.get('swap'):
            continue
        if market.get('quote') != 'USDT':
            continue
        base = market.get('base')
        if not base or base.isdigit():
            continue
        symbols.append(symbol)

    print(f"✅ تعداد نمادهای فیوچرز LBank: {len(symbols)}")
    return symbols


def get_data(symbol, timeframe):
    exchange = get_exchange()
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=1000)
    except Exception as e:
        return None, str(e)

    if not raw or len(raw) < MIN_BARS:
        return None, "not_enough_bars"

    df = pd.DataFrame(raw, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('ts')[['open', 'high', 'low', 'close']]
    df = df.astype(float)

    if timeframe == '1h':
        df_res = df.dropna()
    else:
        df_res = df.resample(timeframe).agg({
            'open':  'first',
            'high':  'max',
            'low':   'min',
            'close': 'last',
        }).dropna()

    return df_res, None


# ---------- محاسبه شباهت پین‌بار با فاصله اقلیدسی ----------
def pin_bar_similarity(row):
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    rng = h - l
    if rng <= 0:
        return None

    body        = abs(c - o)
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l

    v = np.array([body / rng, upper_wick / rng, lower_wick / rng])
    dist = float(np.linalg.norm(v - IDEAL_VECTOR))
    similarity = (1.0 - dist / MAX_DIST) * 100.0

    return {
        'similarity': similarity,
        'body_pct':   body / rng * 100.0,
        'uw_pct':     upper_wick / rng * 100.0,
        'lw_pct':     lower_wick / rng * 100.0,
    }


# ---------- تلگرام ----------
def send_telegram_message(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ توکن یا chat_id تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, data=payload, timeout=15)
        if r.status_code != 200:
            print(f"⚠️ خطا در ارسال پیام: {r.text}")
        else:
            print("✅ پیام به تلگرام ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")


# ---------- اسکن یک نماد ----------
def scan_symbol(sym, tf, stats):
    found = []
    df, err = get_data(sym, tf)
    if df is None:
        if err == "not_enough_bars":
            stats['not_enough_bars'] += 1
        else:
            stats['data_error'] += 1
        return found

    if len(df) < MIN_BARS:
        stats['not_enough_bars'] += 1
        return found

    row = df.iloc[-2]
    row_time = df.index[-2]

    result = pin_bar_similarity(row)
    if result is None:
        stats['not_enough_bars'] += 1
        return found

    if result['similarity'] >= MIN_SIMILARITY:
        found.append({
            'symbol':     sym,
            'timeframe':  tf,
            'time':       row_time.strftime('%Y-%m-%d %H:%M'),
            'o': float(row['open']),
            'h': float(row['high']),
            'l': float(row['low']),
            'c': float(row['close']),
            'sim':        result['similarity'],
            'body_pct':   result['body_pct'],
            'uw_pct':     result['uw_pct'],
            'lw_pct':     result['lw_pct'],
        })
        stats['passed'] += 1

    return found


# ---------- اجرای اصلی ----------
now_utc = pd.Timestamp.utcnow().tz_localize(None)
print(f"🕒 زمان فعلی UTC: {now_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"⭐ اسکن پین‌بار صعودی روزانه (فاصله اقلیدسی)")
print(f"🎯 حداقل شباهت: {MIN_SIMILARITY}%")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
symbols = get_lbank_futures_symbols()
if not symbols:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

signals = []
stats = {
    'total':            len(symbols),
    'data_error':       0,
    'not_enough_bars':  0,
    'passed':           0,
}

for sym in tqdm(symbols, desc="اسکن پین‌بار روزانه"):
    for tf in TIMEFRAMES:
        try:
            found = scan_symbol(sym, tf, stats)
            if found:
                signals.extend(found)
        except Exception:
            stats['data_error'] += 1
            continue
        time.sleep(0.15)

# ---------- ساخت متن و ارسال به تلگرام ----------
if signals:
    signals.sort(key=lambda s: s['sim'], reverse=True)

    lines = []
    lines.append("⭐ <b>پین‌بار صعودی روزانه (فاصله اقلیدسی)</b>")
    lines.append(f"🎯 حداقل شباهت: {MIN_SIMILARITY}%\n")

    for s in signals:
        lines.append(
            f"━━━━━━━━━━━━━━━\n"
            f"🌟 <b>{s['symbol']}</b> [<b>{s['timeframe']}</b>]\n"
            f"🕒 کندل: {s['time']}\n"
            f"📊 O={s['o']:.6g} H={s['h']:.6g} "
            f"L={s['l']:.6g} C={s['c']:.6g}\n"
            f"🎯 شباهت: <b>{s['sim']:.1f}%</b>\n"
            f"   بدنه={s['body_pct']:.1f}% | "
            f"سایه بالا={s['uw_pct']:.1f}% | "
            f"سایه پایین={s['lw_pct']:.1f}%"
        )

    lines.append(f"\n📅 اسکن‌شده: {stats['total']} نماد")
    lines.append(f"⭐ سیگنال‌ها: {len(signals)}")
    lines.append(f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    message = "\n".join(lines)
    send_telegram_message(message)
    print("\n" + message)
else:
    msg = (
        f"⭐ اسکن پین‌بار صعودی روزانه انجام شد — چیزی یافت نشد.\n"
        f"📊 آمار:\n"
        f"  • کل نمادها: {stats['total']}\n"
        f"  • خطای داده: {stats['data_error']}\n"
        f"  • کندل ناکافی: {stats['not_enough_bars']}\n"
        f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    send_telegram_message(msg)
    print("\n" + msg)

print("\n✅ اسکن کامل شد!")
