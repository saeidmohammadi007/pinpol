import os
import pandas as pd
import numpy as np
import ccxt
import time
import requests
from tqdm import tqdm

# ---------- تنظیمات A+ ----------
TIMEFRAMES      = ['1h', '4h']   # ← هر دو تایم‌فریم اسکن می‌شوند
EMA_FAST        = 50
EMA_SLOW        = 200
WICK_RATIO      = 2.5
BODY_MAX_RATIO  = 0.33
CLOSE_ZONE      = 0.25
PULLBACK_TOL    = 0.008
MIN_BARS        = 220
EMA_SLOPE_LOOKBACK = 3
MIN_RR          = 2.0
MIN_TREND_GAP   = 0.002
ENABLE_LONG     = True
ENABLE_SHORT    = True
LOG_CSV         = "signals_log.csv"

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
    """دریافت داده خام 1h و ریسمپل به تایم‌فریم مورد نظر"""
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


# ---------- اندیکاتورها ----------
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
    if lower_wick < WICK_RATIO * max(body, rng * 0.01):
        return False
    if lower_wick < WICK_RATIO * upper_wick:
        return False
    if (c - l) / rng < (1 - CLOSE_ZONE):
        return False
    return True


def is_bear_pin_bar(row):
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    if body / rng > BODY_MAX_RATIO:
        return False
    if upper_wick < WICK_RATIO * max(body, rng * 0.01):
        return False
    if upper_wick < WICK_RATIO * lower_wick:
        return False
    if (h - c) / rng < (1 - CLOSE_ZONE):
        return False
    return True


def near_ema50(row, tol=PULLBACK_TOL):
    ema = row['ema_fast']
    if ema <= 0 or not np.isfinite(ema):
        return False
    return abs(row['low'] - ema) / ema <= tol


def near_ema50_short(row, tol=PULLBACK_TOL):
    ema = row['ema_fast']
    if ema <= 0 or not np.isfinite(ema):
        return False
    return abs(row['high'] - ema) / ema <= tol


def is_uptrend(row):
    if not (np.isfinite(row['ema_fast']) and np.isfinite(row['ema_slow'])):
        return False
    if row['ema_fast'] <= row['ema_slow']:
        return False
    if row['close'] <= row['ema_slow']:
        return False
    if (row['ema_fast'] - row['ema_slow']) / row['ema_slow'] < MIN_TREND_GAP:
        return False
    return True


def is_downtrend(row):
    if not (np.isfinite(row['ema_fast']) and np.isfinite(row['ema_slow'])):
        return False
    if row['ema_fast'] >= row['ema_slow']:
        return False
    if row['close'] >= row['ema_slow']:
        return False
    if (row['ema_slow'] - row['ema_fast']) / row['ema_slow'] < MIN_TREND_GAP:
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


def ema50_falling(df, lookback=EMA_SLOPE_LOOKBACK):
    if len(df) < lookback + 2:
        return False
    now_val  = df['ema_fast'].iloc[-2]
    prev_val = df['ema_fast'].iloc[-2 - lookback]
    if not (np.isfinite(now_val) and np.isfinite(prev_val)):
        return False
    return now_val < prev_val


def has_pullback_before_long(df, lookback=10):
    if len(df) < lookback + 3:
        return False
    seg = df.iloc[-3 - lookback:-2]
    if seg.empty:
        return False
    touched = ((seg['low'] - seg['ema_fast']).abs() / seg['ema_fast'] < 0.015).any()
    return bool(touched)


def has_pullback_before_short(df, lookback=10):
    if len(df) < lookback + 3:
        return False
    seg = df.iloc[-3 - lookback:-2]
    if seg.empty:
        return False
    touched = ((seg['high'] - seg['ema_fast']).abs() / seg['ema_fast'] < 0.015).any()
    return bool(touched)


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


# ---------- اسکن یک نماد در یک تایم‌فریم ----------
def scan_symbol(sym, tf, stats):
    """برمی‌گرداند لیست سیگنال‌های یافت‌شده برای این نماد در این تایم‌فریم"""
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

    df = add_emas(df)
    row = df.iloc[-2]
    row_time = df.index[-2]

    # ---- Long ----
    if ENABLE_LONG:
        if is_uptrend(row):
            if ema50_rising(df):
                if near_ema50(row):
                    if is_bull_pin_bar(row):
                        if has_pullback_before_long(df):
                            entry = float(row['high'])
                            stop  = float(row['low'])
                            risk  = entry - stop
                            if risk > 0:
                                tp1 = entry + risk * 1.0
                                tp2 = entry + risk * 2.0
                                rr = (tp2 - entry) / risk
                                if rr >= MIN_RR:
                                    found.append({
                                        'symbol': sym, 'side': 'LONG',
                                        'timeframe': tf,
                                        'time': row_time.strftime('%Y-%m-%d %H:%M'),
                                        'o': float(row['open']),
                                        'h': float(row['high']),
                                        'l': float(row['low']),
                                        'c': float(row['close']),
                                        'ema50': float(row['ema_fast']),
                                        'ema200': float(row['ema_slow']),
                                        'entry': entry, 'stop': stop,
                                        'tp1': tp1, 'tp2': tp2,
                                    })
                                    stats['passed'] += 1
                                else:
                                    stats['rr'] += 1
                        else:
                            stats['prev_pullback'] += 1
                    else:
                        stats['pinbar'] += 1
                else:
                    stats['pullback'] += 1
            else:
                stats['slope'] += 1
        else:
            stats['trend'] += 1

    # ---- Short ----
    if ENABLE_SHORT:
        if is_downtrend(row):
            if ema50_falling(df):
                if near_ema50_short(row):
                    if is_bear_pin_bar(row):
                        if has_pullback_before_short(df):
                            entry = float(row['low'])
                            stop  = float(row['high'])
                            risk  = stop - entry
                            if risk > 0:
                                tp1 = entry - risk * 1.0
                                tp2 = entry - risk * 2.0
                                rr = (entry - tp2) / risk
                                if rr >= MIN_RR:
                                    found.append({
                                        'symbol': sym, 'side': 'SHORT',
                                        'timeframe': tf,
                                        'time': row_time.strftime('%Y-%m-%d %H:%M'),
                                        'o': float(row['open']),
                                        'h': float(row['high']),
                                        'l': float(row['low']),
                                        'c': float(row['close']),
                                        'ema50': float(row['ema_fast']),
                                        'ema200': float(row['ema_slow']),
                                        'entry': entry, 'stop': stop,
                                        'tp1': tp1, 'tp2': tp2,
                                    })
                                    stats['passed'] += 1

    return found


# ---------- اجرای اصلی ----------
now_utc = pd.Timestamp.utcnow().tz_localize(None)
print(f"🕒 زمان فعلی UTC: {now_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"⭐ اسکن A+ روی تایم‌فریم‌های: {', '.join(TIMEFRAMES)}")
print(f"   Long={ENABLE_LONG} Short={ENABLE_SHORT}")

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
    'trend':            0,
    'slope':            0,
    'pullback':         0,
    'pinbar':           0,
    'prev_pullback':    0,
    'rr':               0,
    'passed':           0,
}

for sym in tqdm(symbols, desc="اسکن A+"):
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
    lines.append(f"⭐ <b>ستاپ‌های A+ — پین‌بار در پولبک EMA{EMA_FAST}</b>")
    lines.append(f"⏱ تایم‌فریم‌ها: {' + '.join(TIMEFRAMES)}")
    lines.append(f"⚙️ سایه≥{WICK_RATIO}x | بدنه≤{BODY_MAX_RATIO} | "
                 f"پولبک≤{PULLBACK_TOL*100}%\n")

    # مرتب‌سازی: ابتدا 1h، سپس 4h (یا هر ترتیب TIMEFRAMES)
    for tf in TIMEFRAMES:
        tf_signals = [s for s in signals if s['timeframe'] == tf]
        if not tf_signals:
            continue

        lines.append(f"\n🔹 <b>تایم‌فریم {tf}</b> ({len(tf_signals)} سیگنال)")
        for s in tf_signals:
            icon = "🟢" if s['side'] == 'LONG' else "🔴"
            lines.append(
                f"━━━━━━━━━━━━━━━\n"
                f"🌟 <b>{s['symbol']}</b> — {icon} {s['side']} "
                f"[<b>{s['timeframe']}</b>]\n"
                f"🕒 کندل: {s['time']}\n"
                f"📊 OHLC: O={s['o']:.6g} H={s['h']:.6g} "
                f"L={s['l']:.6g} C={s['c']:.6g}\n"
                f"📈 EMA50={s['ema50']:.6g} | EMA200={s['ema200']:.6g}\n"
                f"🎯 ورود: {s['entry']:.6g}\n"
                f"🛑 استاپ: {s['stop']:.6g}\n"
                f"✅ TP1 (1R): {s['tp1']:.6g}\n"
                f"✅ TP2 (2R): {s['tp2']:.6g}"
            )

    lines.append(f"\n📅 اسکن‌شده: {stats['total']} نماد × {len(TIMEFRAMES)} تایم‌فریم")
    lines.append(f"⭐ سیگنال‌ها: {len(signals)}")
    lines.append(f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")

    message = "\n".join(lines)
    send_telegram_message(message)
    print("\n" + message)
else:
    msg = (
        f"⭐ اسکن A+ انجام شد — هیچ ستاپی یافت نشد.\n"
        f"📊 آمار فیلترها:\n"
        f"  • کل نمادها: {stats['total']} × {len(TIMEFRAMES)} تایم‌فریم\n"
        f"  • خطای داده: {stats['data_error']}\n"
        f"  • کندل ناکافی: {stats['not_enough_bars']}\n"
        f"  • روند نامناسب: {stats['trend']}\n"
        f"  • شیب EMA50: {stats['slope']}\n"
        f"  • پولبک به EMA50: {stats['pullback']}\n"
        f"  • پین‌بار ضعیف: {stats['pinbar']}\n"
        f"  • اصلاح قبل پین‌بار: {stats['prev_pullback']}\n"
        f"  • RR ناکافی: {stats['rr']}\n"
        f"🕒 {now_utc.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    send_telegram_message(msg)
    print("\n" + msg)

print("\n✅ اسکن کامل شد!")
