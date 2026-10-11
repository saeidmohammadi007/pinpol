import os
import pandas as pd
import numpy as np
import ccxt
import time
import requests
from tqdm import tqdm

# ---------- تنظیمات ----------
TIMEFRAMES      = ['1h', '4h']
WICK_RATIO      = 2.5
BODY_MAX_RATIO  = 0.33
CLOSE_ZONE      = 0.25
MIN_BARS        = 100
LOG_CSV         = "bullish_pinbar_log.csv"

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


# ---------- تشخیص پین‌بار صعودی ----------
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
    if lower_wick < WICK_RATIO * max(body, rng * 0.01):
        return False
    if lower_wick < WICK_RATIO * upper_wick:
        return False
    if (c - l) / rng < (1 - CLOSE_ZONE):
        return False
    return True


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

    if is_bull_pin_bar(row):
        found.append({
            'symbol': sym,
            'timeframe': tf,
            'time': row_time.strftime('%Y-%m-%d %H:%M'),
            'o': float(row['open']),
            'h': float(row['high']),
            'l': float(row['low']),
            'c': float(row['close']),
        })
        stats['passed'] += 1

    return found


# ---------- اجرای اصلی ----------
now_utc = pd.Timestamp.utcnow().tz_localize(None)
print(f"🕒 زمان فعلی UTC: {now_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"⭐ اسکن پین‌بار صعودی روی: {', '.join(TIMEFRAMES)}")

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

for sym in tqdm(symbols, desc="اسکن پین‌بار صعودی"):
    for tf in TIMEFRAMES:
        try:
            found = scan_symbol(sym, tf, stats)
            if found:
                signals.extend(found)
        except Exception:
            stats['data_error'] += 1
            continue
        time.sleep(0.15)

# ---------- ذخیره CSV ----------
if signals:
    try:
        df_log = pd.DataFrame(signals)
        if os.path.exists(LOG_CSV):
            df_log.to_csv(LOG_CSV, mode='a', header=False, index=False)
        else:
            df_log.to_csv(LOG_CSV, index=False)
        print(f"💾 {len(signals)} سیگنال در {LOG_CSV} ذخیره شد.")
    except Exception as e:
        print(f"⚠️ خطا در ذخیره CSV: {e}")

# ---------- ارسال به تلگرام ----------
if signals:
    lines = []
    lines.append("⭐ <b>پین‌بار صعودی</b>")
    lines.append(f"⏱ تایم‌فریم‌ها: {' + '.join(TIMEFRAMES)}")
    lines.append(f"⚙️ سایه≥{WICK_RATIO}x | بدنه≤{BODY_MAX_RATIO}\n")

    for tf in TIMEFRAMES:
        tf_signals = [s for s in signals if s['timeframe'] == tf]
        if not tf_signals:
            continue

        lines.append(f"\n🔹 <b>تایم‌فریم {tf}</b> ({len(tf_signals)} سیگنال)")
        for s in tf_signals:
            lines.append(
                f"━━━━━━━━━━━━━━━\n"
                f"🌟 <b>{s['symbol']}</b> [<b>{s['timeframe']}</b>]\n"
                f"🕒 کندل: {s['time']}\n"
                f"📊 O={s['o']:.6g} H={s['h']:.6g} "
                f"L={s['l']:.6g} C={s['c']:.6g}"
            )

    lines.append(f"\n📅 اسکن‌شده: {stats['total']} نماد × {len(TIMEFRAMES)} تایم‌فریم")
    lines.append(f"⭐ سیگنال‌ها: {len(signals)}")
    lines.append(f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    message = "\n".join(lines)
    send_telegram_message(message)
    print("\n" + message)
else:
    msg = (
        f"⭐ اسکن پین‌بار صعودی انجام شد — چیزی یافت نشد.\n"
        f"📊 آمار:\n"
        f"  • کل نمادها: {stats['total']} × {len(TIMEFRAMES)} تایم‌فریم\n"
        f"  • خطای داده: {stats['data_error']}\n"
        f"  • کندل ناکافی: {stats['not_enough_bars']}\n"
        f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    send_telegram_message(msg)
    print("\n" + msg)

print("\n✅ اسکن کامل شد!")
