#!/usr/bin/env python3
# bot_v12.py
# Binance Spot CCXT scalping bot (multi-bot friendly) with:
# - Configurable symbol/timeframe
# - Maker-first execution (LIMIT postOnly) with fast fallback to MARKET
# - Spread filter + anti-fee net-profit viability guard (fee + slippage + spread)
# - Cooldown caps: any "pause" is capped at 60s (per user requirement)
# - Trailing take-profit + optional time-exit per position
# - Telegram alerts on BUY/SELL/WIN/LOSS/ERROR/HEARTBEAT (never crashes bot)
# - Rotating logs + persistent state + trade journal CSV
# - Instance-safe files: state/journal/log file names can be overridden in config,
#   otherwise they are derived from misc.instance_id (or trading.symbol).
#
# Run:
#   python bot.py --config config.json
#
# Notes:
# - This bot is designed for spot. It can be run multiple times in parallel in different folders.
# - Higher frequency => higher fee/noise exposure. Keep fee guard ON.

import argparse
import csv
import json
import logging
import os
import re
def _safe_client_order_id(s: str, max_len: int = 36) -> str:
    s = re.sub(r'[^A-Za-z0-9_-]', '', str(s))
    return s[:max_len] if s else ''

import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional, Tuple

import ccxt


# ---------- colors / console formatting ----------
class C:
    RESET = "\033[0m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"


def _supports_ansi() -> bool:
    return sys.stdout.isatty()


try:
    from colorama import init as _colorama_init  # type: ignore

    _colorama_init(autoreset=True)
    _COLORAMA = True
except Exception:
    _COLORAMA = False


class ConsoleFormatter(logging.Formatter):
    TAG_COLORS = {
        "INFO": C.CYAN,
        "OK": C.GREEN,
        "WARN": C.YELLOW,
        "ERROR": C.RED,
        "BUY": C.GREEN,
        "ADD": C.GREEN,
        "SELL": C.MAGENTA,
        "WIN": C.GREEN,
        "LOSS": C.RED,
        "STATE": C.GRAY,
    }

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover
        tag = getattr(record, "tag", record.levelname)
        color = self.TAG_COLORS.get(tag, C.GRAY)
        ts = datetime.fromtimestamp(record.created, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        msg = record.getMessage()
        prefix = f"[{tag}]"
        if (_supports_ansi() or _COLORAMA) and color:
            prefix = f"{color}{prefix}{C.RESET}"
        return f"{prefix} {ts} | {msg}"


def build_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(log_file)  # unique per instance
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ConsoleFormatter())
    logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_fmt = logging.Formatter("[%(levelname)s] %(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S UTC")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    return logger


def ts_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------- utility helpers ----------
def safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period or period <= 0:
        return None
    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = (v * k) + (ema_val * (1 - k))
    return ema_val


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _validated_percent(value: Any, default: float, name: str, log_fn: Callable[[str, str], None]) -> float:
    v = safe_float(value, None)
    if v is None or v < 0:
        log_fn("WARN", f"Invalid {name} ({value}); using default {default}.")
        return default
    return float(v)


def ensure_journal_header(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "time_utc",
                "symbol",
                "action",
                "reason_code",
                "reason_text",
                "slot_id",
                "qty",
                "price",
                "notional",
                "fee_paid",
                "fee_currency",
                "pnl_net_quote",
                "pnl_pct_net",
                "duration_s",
                "client_id",
                "candle_ts",
                "execution",
            ]
        )


def journal_row(path: str, row: List[Any]) -> None:
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    except Exception:
        # keep bot alive
        pass


def amount_to_precision(exchange, symbol: str, amount: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)


def price_to_precision(exchange, symbol: str, price: float) -> float:
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return float(price)


def market_min_cost(market: Dict[str, Any]) -> Optional[float]:
    limits = market.get("limits") or {}
    return safe_float((limits.get("cost") or {}).get("min"), None)


def market_min_amount(market: Dict[str, Any]) -> Optional[float]:
    limits = market.get("limits") or {}
    return safe_float((limits.get("amount") or {}).get("min"), None)


def fetch_with_retry(fn: Callable, log_fn: Callable[[str, str], None], attempts: int = 3, delay_seconds: float = 1.5):
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            last_err = e
            log_fn("WARN", f"Retry {i+1}/{attempts} after error: {e}")
            time.sleep(delay_seconds * (1.5 ** i))
    if last_err:
        raise last_err


def parse_symbol_assets(symbol: str, cfg: Dict[str, Any]) -> Tuple[str, str]:
    b = (cfg.get("trading") or {}).get("base_asset")
    q = (cfg.get("trading") or {}).get("quote_asset")
    if b and q:
        return str(b), str(q)
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return base, quote
    raise ValueError("Cannot infer base/quote. Provide trading.base_asset and trading.quote_asset.")


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("/", "_")
    s = re.sub(r"[^a-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "bot"


# ---------- telegram alerts (optional, never crash bot) ----------
def tg_send(cfg: Dict[str, Any], text: str) -> None:
    tg = (cfg.get("alerts") or {}).get("telegram") or {}
    if not tg.get("enabled"):
        return
    token = tg.get("bot_token")
    chat_id = tg.get("chat_id")
    if not token or not chat_id:
        return
    try:
        import requests  # type: ignore

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, timeout=6, json={"chat_id": chat_id, "text": text})
    except Exception:
        return


# ---------- exchange ----------
def make_exchange(cfg: Dict[str, Any]):
    ex_id = (cfg.get("exchange") or {}).get("id", "binance")
    klass = getattr(ccxt, ex_id)
    params = {
        "apiKey": (cfg.get("exchange") or {}).get("api_key"),
        "secret": (cfg.get("exchange") or {}).get("api_secret"),
        "enableRateLimit": True,
        "timeout": 30000,
        "options": {
            "defaultType": (cfg.get("exchange") or {}).get("market_type", "spot"),
            "adjustForTimeDifference": bool((cfg.get("exchange") or {}).get("adjust_for_time_difference", True)),
        },
    }
    if (cfg.get("exchange") or {}).get("password"):
        params["password"] = (cfg.get("exchange") or {})["password"]
    ex = klass(params)
    if (cfg.get("exchange") or {}).get("sandbox", False) and hasattr(ex, "set_sandbox_mode"):
        ex.set_sandbox_mode(True)
    return ex


def fetch_free_balances(exchange, base: str, quote: str) -> Tuple[float, float]:
    bal = exchange.fetch_balance()
    free_base = safe_float((bal.get(base) or {}).get("free"), 0.0) or 0.0
    free_quote = safe_float((bal.get(quote) or {}).get("free"), 0.0) or 0.0
    return free_base, free_quote


def estimate_net_pnl_quote(entry_price: float, exit_price: float, base_amount: float, fee_rate: float) -> float:
    gross = (exit_price - entry_price) * base_amount
    fees = (entry_price * base_amount + exit_price * base_amount) * fee_rate
    return gross - fees


# ---------- client ids ----------
def deterministic_client_id(instance_id: str, symbol: str, action: str, candle_ts: int, seq: int, slot_id: int) -> str:
    base = symbol.replace("/", "")
    inst = slugify(instance_id)[:16]
    return f"{inst}-{base}-{action}-S{slot_id}-{candle_ts}-{seq}"


# ---------- execution helpers ----------
def get_best_bid_ask(exchange, symbol: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        ob = exchange.fetch_order_book(symbol, limit=5)
        bid = ob["bids"][0][0] if ob.get("bids") else None
        ask = ob["asks"][0][0] if ob.get("asks") else None
        return (float(bid) if bid else None, float(ask) if ask else None)
    except Exception:
        return (None, None)


def spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 0.0
    return ((ask - bid) / mid) * 10000.0


def place_limit_postonly(exchange, symbol: str, side: str, amount: float, price: float, dry_run: bool, params: Dict[str, Any]) -> Dict[str, Any]:
    if dry_run:
        return {"id": f"DRY_LIMIT_{side.upper()}", "filled": 0.0, "average": None, "status": "open"}
    return exchange.create_order(symbol, "limit", side, amount, price, params)


def place_market(exchange, symbol: str, side: str, amount: float, dry_run: bool, params: Dict[str, Any]) -> Dict[str, Any]:
    if dry_run:
        t = exchange.fetch_ticker(symbol)
        px = float(t["last"])
        return {"id": f"DRY_MKT_{side.upper()}", "filled": amount, "average": px, "status": "closed"}
    return exchange.create_order(symbol, "market", side, amount, None, params)


def cancel_order_safely(exchange, order_id: str, symbol: str) -> None:
    try:
        exchange.cancel_order(order_id, symbol)
    except Exception:
        pass


def fetch_order_safely(exchange, order_id: str, symbol: str) -> Optional[Dict[str, Any]]:
    try:
        return exchange.fetch_order(order_id, symbol)
    except Exception:
        return None


def collect_fee_from_order_or_trades(exchange, symbol: str, order: Dict[str, Any], since_ms: int) -> Tuple[float, str]:
    # Best-effort. Returns (fee_cost, fee_currency). If unknown, returns (0.0,"").
    try:
        fee = order.get("fee")
        if fee and fee.get("cost") is not None:
            return float(fee["cost"]), str(fee.get("currency") or "")
    except Exception:
        pass

    oid = order.get("id")
    if not oid:
        return (0.0, "")
    try:
        trades = exchange.fetch_my_trades(symbol, since=since_ms, limit=50)
        fee_cost = 0.0
        fee_ccy = ""
        for t in trades:
            if str(t.get("order")) != str(oid):
                continue
            f = t.get("fee") or {}
            c = safe_float(f.get("cost"), 0.0) or 0.0
            cur = str(f.get("currency") or "")
            fee_cost += c
            if not fee_ccy and cur:
                fee_ccy = cur
        return (float(fee_cost), fee_ccy)
    except Exception:
        return (0.0, "")


def execute_order(
    exchange,
    symbol: str,
    side: str,
    amount: float,
    client_id: str,
    execution_cfg: Dict[str, Any],
    dry_run: bool,
    log_fn: Callable[[str, str], None],
) -> Tuple[Dict[str, Any], str]:
    """
    Returns (final_order, execution_label)
    execution_label: "maker" or "market"
    """
    mode = execution_cfg.get("execution_mode", "maker_first")
    maker_timeout_s = int(execution_cfg.get("maker_timeout_seconds", 10))
    maker_poll_s = float(execution_cfg.get("maker_poll_seconds", 1))
    fallback_to_market = bool(execution_cfg.get("fallback_to_market", True))
    max_fallback_slippage_bps = float(execution_cfg.get("max_fallback_slippage_bps", 12.0))
    price_offset_bps = float(execution_cfg.get("maker_price_offset_bps", 0.0))

# XRP: do NOT send newClientOrderId (Binance rejects it)
params_common = {}


    if mode == "market_only":
        return (fetch_with_retry(lambda: place_market(exchange, symbol, side, amount, dry_run, params_common), log_fn), "market")

        bid, ask = get_best_bid_ask(exchange, symbol)
    if bid is None or ask is None:
        return (fetch_with_retry(lambda: place_market(exchange, symbol, side, amount, dry_run, params_common), log_fn), "market")

    # choose maker price that stays on maker side
    if side == "buy":
        px = min(ask * (1 - 0.0001), bid * (1 + price_offset_bps / 10000.0))
    else:
        px = max(bid * (1 + 0.0001), ask * (1 - price_offset_bps / 10000.0))

    px = price_to_precision(exchange, symbol, px)

    maker_params = dict(params_common)
    maker_params.update({"timeInForce": "GTX", "postOnly": True})

    try:
        order = fetch_with_retry(lambda: place_limit_postonly(exchange, symbol, side, amount, px, dry_run, maker_params), log_fn)
    except ccxt.ExchangeError as e:
        log_fn("WARN", f"Maker order rejected ({side}) @ {px}: {e}")
        if fallback_to_market:
            return (fetch_with_retry(lambda: place_market(exchange, symbol, side, amount, dry_run, params_common), log_fn), "market")
        raise

    oid = order.get("id")
    if not oid:
        return (order, "maker")

    start = time.time()
    while time.time() - start < maker_timeout_s:
        o = fetch_order_safely(exchange, oid, symbol)
        if o:
            status = str(o.get("status") or "").lower()
            filled = safe_float(o.get("filled"), 0.0) or 0.0
            remaining = safe_float(o.get("remaining"), None)
            if status == "closed" or (remaining is not None and remaining <= 0):
                return (o, "maker")
            if filled > 0 and remaining is not None and remaining <= 0:
                return (o, "maker")
        time.sleep(maker_poll_s)

    cancel_order_safely(exchange, oid, symbol)

    if not fallback_to_market:
        o = fetch_order_safely(exchange, oid, symbol)
        return (o or order, "maker")

    bid2, ask2 = get_best_bid_ask(exchange, symbol)
    if bid2 and ask2:
        mid = (bid2 + ask2) / 2
        ref = ask2 if side == "buy" else bid2
        if mid > 0:
            move_bps = abs((ref - mid) / mid) * 10000.0
            if move_bps > max_fallback_slippage_bps:
                raise RuntimeError(f"Fallback slippage too high: {move_bps:.1f} bps > {max_fallback_slippage_bps}")

    return (fetch_with_retry(lambda: place_market(exchange, symbol, side, amount, dry_run, params_common), log_fn), "market")


# ---------- sizing ----------
def compute_dynamic_entry_quote(cfg: Dict[str, Any], free_quote: float, open_positions: int) -> float:
    trading = cfg.get("trading") or {}
    pm = cfg.get("position_management") or {}
    max_positions = int(pm.get("max_positions", 1))
    exposure_pct = float(pm.get("max_total_exposure_percent_of_free", 0.90))
    exposure_pct = clamp(exposure_pct, 0.0, 1.0)

    mode = trading.get("budget_mode", "fixed_quote")
    fixed = float(trading.get("trade_quote_amount_usdc", 48.0))
    pct = float(trading.get("trade_quote_percent_free", 0.35))
    compounding = bool(trading.get("compounding_enabled", False))

    remaining_slots = max(0, max_positions - open_positions)
    if remaining_slots <= 0:
        return 0.0

    cap_total = free_quote * exposure_pct
    cap_per_slot = cap_total / max(1, max_positions)

    if not compounding:
        want = fixed
        quote_cap = min(want, cap_per_slot, free_quote)
        if quote_cap < fixed:
            return 0.0
    else:
        if mode == "percent_free":
            want = free_quote * clamp(pct, 0.0, 1.0)
        else:
            want = fixed
        quote_cap = min(want, cap_per_slot, free_quote)

    return max(0.0, quote_cap)


def atr_percent(ohlcv: List[List[float]], period: int) -> Optional[float]:
    if len(ohlcv) < period + 2:
        return None
    trs: List[float] = []
    closed = ohlcv[:-1]
    start = max(1, len(closed) - period)
    for i in range(start, len(closed)):
        high = float(closed[i][2])
        low = float(closed[i][3])
        prev_close = float(closed[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return None
    atr = sum(trs) / len(trs)
    last_close = float(closed[-1][4])
    if last_close == 0:
        return None
    return (atr / last_close) * 100.0


# ---------- viability guard ----------
def profit_viable(
    entry_price: float,
    take_profit_pct: float,
    fee_rate: float,
    slippage_buffer_pct: float,
    spread_buffer_pct: float,
    min_net_profit_pct: float,
) -> bool:
    target_price = entry_price * (1 + take_profit_pct)
    gross_gain = target_price - entry_price
    est_fees = (entry_price + target_price) * fee_rate
    slippage = entry_price * slippage_buffer_pct
    spread = entry_price * spread_buffer_pct
    net = gross_gain - est_fees - slippage - spread
    return (net / entry_price) >= min_net_profit_pct


# ---------- instance paths ----------
def resolve_paths(cfg: Dict[str, Any]) -> Tuple[str, str, str]:
    runtime = cfg.get("runtime") or {}
    misc = cfg.get("misc") or {}
    symbol = (cfg.get("trading") or {}).get("symbol", "PAIR")

    instance_id = str(misc.get("instance_id") or misc.get("bot_name") or symbol)
    sid = slugify(instance_id)
    log_dir = str(runtime.get("log_dir") or "logs")
    os.makedirs(log_dir, exist_ok=True)

    state_file = str(runtime.get("state_file") or f"state_{sid}.json")
    journal_file = str(runtime.get("journal_file") or f"trade_journal_{sid}.csv")
    log_file = str(runtime.get("log_file") or os.path.join(log_dir, f"bot_{sid}.log"))

    return state_file, journal_file, log_file


def validate_config(cfg: Dict[str, Any]) -> None:
    required_sections = ["exchange", "trading", "strategy", "risk", "fees", "guardrails", "execution", "runtime", "misc"]
    for sec in required_sections:
        if sec not in cfg:
            raise ValueError(f"Missing config section: {sec}")

    trading = cfg.get("trading") or {}
    strategy = cfg.get("strategy") or {}
    risk = cfg.get("risk") or {}
    runtime = cfg.get("runtime") or {}

    if not trading.get("symbol"):
        raise ValueError("Config trading.symbol is required")

    for key in ["ma_fast_period", "ma_mid_period", "ma_slow_period"]:
        if safe_float(strategy.get(key), None) is None:
            raise ValueError(f"Config strategy.{key} must be numeric")

    for key in ["stop_loss_percent", "take_profit_percent"]:
        if safe_float(risk.get(key), None) is None:
            raise ValueError(f"Config risk.{key} must be numeric")

    if int(runtime.get("poll_seconds", 1)) <= 0:
        raise ValueError("Config runtime.poll_seconds must be > 0")


# ---------- state (multi-position) ----------
def _default_position(slot_id: int) -> Dict[str, Any]:
    return {
        "slot_id": slot_id,
        "open": False,
        "entry_price": None,
        "base_amount": None,
        "highest_price": None,
        "breakeven_armed": False,
        "adds_count": 0,
        "entry_quote_spent_est": 0.0,
        "opened_time_ms": 0,
        "last_action_ms": 0,
        "last_client_id": "",
        "last_execution": "",
        "entry_fee_paid": 0.0,
        "entry_fee_currency": "",
    }


def load_state(state_file: str, slots: int) -> Dict[str, Any]:
    slots = max(1, int(slots))
    if os.path.exists(state_file):
        try:
            st = load_json(state_file)
            if "positions" not in st or not isinstance(st["positions"], list):
                st["positions"] = []
            positions = st["positions"][:slots]
            while len(positions) < slots:
                positions.append(_default_position(len(positions) + 1))
            st["positions"] = positions
            st.setdefault("start_time_ms", int(time.time() * 1000))
            st.setdefault("last_processed_candle_ts", 0)
            st.setdefault("last_order_ts_ms", 0)
            st.setdefault("trade_seq", 0)
            st.setdefault("realized_profit_quote_net", 0.0)
            st.setdefault("last_entry_time_ms", 0)
            st.setdefault("consecutive_losses", 0)
            st.setdefault("cooldown_until_ms", 0)
            return st
        except Exception:
            pass

    return {
        "positions": [_default_position(i + 1) for i in range(slots)],
        "start_time_ms": int(time.time() * 1000),
        "last_processed_candle_ts": 0,
        "last_order_ts_ms": 0,
        "trade_seq": 0,
        "realized_profit_quote_net": 0.0,
        "last_entry_time_ms": 0,
        "consecutive_losses": 0,
        "cooldown_until_ms": 0,
    }


def save_state(state_file: str, state: Dict[str, Any]) -> None:
    save_json(state_file, state)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    args = ap.parse_args()

    cfg = load_json(args.config)
    validate_config(cfg)

    # resolve instance-safe paths
    state_file, journal_file, log_file = resolve_paths(cfg)
    logger = build_logger(log_file)

    def log(tag: str, msg: str) -> None:
        logger.info(msg, extra={"tag": tag})

    ensure_journal_header(journal_file)

    trading_cfg = cfg.get("trading") or {}
    symbol = str(trading_cfg["symbol"])
    timeframe = str(trading_cfg.get("timeframe", "5m"))
    trade_quote_amount = float(trading_cfg.get("trade_quote_amount_usdc", 48.0))
    compounding = bool(trading_cfg.get("compounding_enabled", False))
    base, quote = parse_symbol_assets(symbol, cfg)
    dry_run = bool((cfg.get("runtime") or {}).get("dry_run", False))

    strat = cfg.get("strategy") or {}
    ma_fast = int(strat.get("ma_fast_period", 5))
    ma_mid = int(strat.get("ma_mid_period", 13))
    ma_slow = int(strat.get("ma_slow_period", 50))
    require_above_ma_mid = bool(strat.get("require_price_above_ma_mid", False))
    max_entry_distance_ma_mid_pct = float(strat.get("max_entry_distance_above_ma_mid_percent", 1.2)) / 100.0
    allow_range_entries = bool(strat.get("allow_range_entries", True))
    range_reentry_cooldown_seconds = int(strat.get("range_reentry_cooldown_seconds", 20))
    range_reentry_cooldown_seconds = min(60, max(0, range_reentry_cooldown_seconds))
    min_fast_slope_bps = float(strat.get("min_fast_slope_bps", 2.0))
    volatility_atr_period = max(1, int(strat.get("volatility_atr_period", 14)))
    min_atr_percent = _validated_percent(strat.get("min_atr_percent", 0.15), 0.15, "min_atr_percent", log)
    max_atr_percent = _validated_percent(strat.get("max_atr_percent", 1.20), 1.20, "max_atr_percent", log)

    # Normalize (percent units)
    min_atr_percent = max(0.0, float(min_atr_percent))
    max_atr_percent = max(0.0, float(max_atr_percent))
    if max_atr_percent > 0 and min_atr_percent > max_atr_percent:
        min_atr_percent, max_atr_percent = max_atr_percent, min_atr_percent

    risk = cfg.get("risk") or {}
    stop_loss_pct_raw = _validated_percent(risk.get("stop_loss_percent", 0.35), 0.35, "stop_loss_percent", log)
    take_profit_pct_raw = _validated_percent(risk.get("take_profit_percent", 0.35), 0.35, "take_profit_percent", log)

    # Convert config "percent units" into fractions
    stop_loss_pct = stop_loss_pct_raw / 100.0
    take_profit_pct = take_profit_pct_raw / 100.0

    max_hold_minutes = max(0, int(risk.get("max_hold_minutes", 6)))
    time_exit_enabled = bool(risk.get("time_exit_enabled", True))

    trailing_cfg = risk.get("trailing_take_profit") or {}
    trailing_enabled = bool(trailing_cfg.get("enabled", True))
    trailing_distance_pct_raw = _validated_percent(trailing_cfg.get("trail_distance_percent", 0.12), 0.12, "trail_distance_percent", log)
    trailing_activate_pct_raw = _validated_percent(trailing_cfg.get("activate_after_profit_percent", 0.20), 0.20, "activate_after_profit_percent", log)
    trailing_distance_pct = trailing_distance_pct_raw / 100.0
    trailing_activate_pct = trailing_activate_pct_raw / 100.0

    cooldown_seconds_after_close = int(risk.get("cooldown_seconds_after_close", 1))
    min_seconds_between_orders = int(risk.get("min_seconds_between_orders", 2))
    min_seconds_between_entries = int(risk.get("min_seconds_between_entries", 3))

    # Enforce the "max 60s pause" rule (user requirement)
    cooldown_seconds_after_close = min(60, max(0, cooldown_seconds_after_close))
    min_seconds_between_entries = min(60, max(0, min_seconds_between_entries))
    min_seconds_between_orders = max(0, min_seconds_between_orders)

    loss_cd_cfg = risk.get("loss_cooldown") or {}
    loss_cd_enabled = bool(loss_cd_cfg.get("enabled", True))
    max_consecutive_losses = int(loss_cd_cfg.get("max_consecutive_losses", 2))
    loss_cooldown_minutes = int(loss_cd_cfg.get("cooldown_minutes", 1))  # default 1m
    # cap to 1 minute effectively
    loss_cooldown_minutes = min(1, max(0, loss_cooldown_minutes))

    fees_cfg = cfg.get("fees") or {}
    fee_rate = float(fees_cfg.get("assumed_fee_rate", 0.001))
    min_net_profit_pct = float(fees_cfg.get("min_net_profit_percent", 0.28)) / 100.0
    breakeven_lock_pct = float(fees_cfg.get("breakeven_lock_percent", 0.10)) / 100.0
    slippage_buffer_pct = float(fees_cfg.get("slippage_buffer_percent", 0.06)) / 100.0

    # Fee coverage guard: if TP can't cover fees+buffers+min_net, bump TP (safer than trading at net-negative)
    # We keep bump small and log it.
    guard_tp_floor_pct = float(fees_cfg.get("tp_floor_percent", 0.0)) / 100.0  # optional explicit floor
    # Compute a conservative floor: round-trip fee + min_net + buffers (in fraction)
    conservative_floor = (2 * fee_rate) + min_net_profit_pct + slippage_buffer_pct + 0.0002  # +0.02% micro-buffer
    if guard_tp_floor_pct > 0:
        conservative_floor = max(conservative_floor, guard_tp_floor_pct)
    if take_profit_pct < conservative_floor:
        old = take_profit_pct
        take_profit_pct = conservative_floor
        take_profit_pct_raw = take_profit_pct * 100.0
        log("WARN", f"TP bumped from {old*100:.2f}% to {take_profit_pct_raw:.2f}% (fee coverage guard).")
        tg_send(cfg, f"⚠️ TP bumped to {take_profit_pct_raw:.2f}% for fee coverage on {symbol}")

    guard = cfg.get("guardrails") or {}
    viability = (guard.get("net_profit_viability") or {})
    viability_enabled = bool(viability.get("enabled", True))
    viability_margin_pct = float(viability.get("extra_margin_percent", 0.05)) / 100.0

    spread_cfg = (guard.get("spread_filter") or {})
    spread_enabled = bool(spread_cfg.get("enabled", True))
    max_spread_bps = max(0.0, safe_float(spread_cfg.get("max_spread_bps", 10.0), 10.0) or 10.0)

    reserve_cfg = (guard.get("fee_reserve") or {})
    reserve_enabled = bool(reserve_cfg.get("enabled", True))
    min_quote_reserve = float(reserve_cfg.get("min_quote_reserve", 2.0))

    execution_cfg = cfg.get("execution") or {}

    runtime = cfg.get("runtime") or {}
    poll_seconds = int(runtime.get("poll_seconds", 2))
    candles_limit = int(runtime.get("candles_limit", 240))
    max_retries = int(runtime.get("max_retries", 3))
    retry_delay = float(runtime.get("retry_delay_seconds", 1.5))
    heartbeat_minutes = int(runtime.get("heartbeat_minutes", 10))

    pm = cfg.get("position_management") or {}
    allow_multi = bool(pm.get("allow_multiple_positions", False))
    max_positions_cfg = max(1, int(pm.get("max_positions", 1)))
    max_positions = max_positions_cfg if allow_multi else 1
    exposure_pct = float(pm.get("max_total_exposure_percent_of_free", 0.90))

    state = load_state(state_file, slots=max_positions)

    exchange = make_exchange(cfg)
    markets = fetch_with_retry(lambda: exchange.load_markets(), log, attempts=max_retries, delay_seconds=retry_delay)
    if symbol not in markets:
        raise RuntimeError(f"Symbol not available: {symbol}")
    market = markets[symbol]
    min_cost = market_min_cost(market)
    min_amount = market_min_amount(market)

    log("INFO", "===== SESSION SUMMARY (V12) =====")
    log("INFO", f"INSTANCE files | state={state_file} journal={journal_file} log={log_file}")
    log(
        "INFO",
        (
            f"PAIR {symbol} @ {timeframe} | max_positions={max_positions} exposure_cap={exposure_pct*100:.0f}% free | "
            f"tradeSize={trade_quote_amount:.2f}{quote} compounding={'on' if compounding else 'off'} | "
            f"ATR range {min_atr_percent:.3f}% - {max_atr_percent:.3f}% (percent units)"
        ),
    )
    log(
        "INFO",
        (
            f"RISK SL={stop_loss_pct*100:.2f}% TP={take_profit_pct*100:.2f}% trail={'on' if trailing_enabled else 'off'} "
            f"cooldown_close<=60s={cooldown_seconds_after_close}s entry_gap<=60s={min_seconds_between_entries}s order_gap={min_seconds_between_orders}s"
        ),
    )
    log(
        "INFO",
        (
            f"GUARDS fee={fee_rate*100:.3f}% minNet={min_net_profit_pct*100:.2f}% viability={'on' if viability_enabled else 'off'} | "
            f"spreadFilter={'on' if spread_enabled else 'off'} maxSpread={max_spread_bps:.1f}bps | reserve={'on' if reserve_enabled else 'off'} minReserve={min_quote_reserve:.2f}{quote}"
        ),
    )
    log("INFO", f"EXEC exec_mode={execution_cfg.get('execution_mode','maker_first')} dry_run={dry_run} heartbeat={heartbeat_minutes}m")
    log("INFO", "===============================")

    tg_send(cfg, f"✅ BOTV12 started\n{symbol} {timeframe}\nTP={take_profit_pct*100:.2f}% SL={stop_loss_pct*100:.2f}%\nCooldown max 60s")

    stats = {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pnl_quote": 0.0,
        "session_start_ms": int(time.time() * 1000),
        "last_heartbeat_ms": 0,
        "last_action_ts_ms": 0,
    }

    def count_open_positions() -> int:
        return sum(1 for p in state["positions"] if p.get("open"))

    last_range_entry_ms = 0

    while True:
        try:
            now_ms = int(time.time() * 1000)

            # short global cooldown (capped to 60s)
            cd_until = int(state.get("cooldown_until_ms", 0))
            if cd_until > now_ms:
                remaining = (cd_until - now_ms) / 1000.0
                log("WARN", f"PAUSED | {remaining:.1f}s left (short cooldown)")
                time.sleep(min(poll_seconds, 2))
                continue

            # heartbeat
            if heartbeat_minutes > 0 and (now_ms - stats["last_heartbeat_ms"]) >= heartbeat_minutes * 60 * 1000:
                uptime_h = (now_ms - stats["session_start_ms"]) / (1000 * 60 * 60)
                winrate = (stats["wins"] / max(1, stats["trades"])) * 100.0
                trades_per_hour = stats["trades"] / max(1e-6, uptime_h)
                last_action_age = (now_ms - stats.get("last_action_ts_ms", now_ms)) / 1000
                msg = (
                    f"📊 HEARTBEAT {symbol}\n"
                    f"uptime={uptime_h:.2f}h trades={stats['trades']} W={stats['wins']} L={stats['losses']} "
                    f"wr={winrate:.1f}% pnl≈{stats['pnl_quote']:.4f} {quote}\n"
                    f"trades/h≈{trades_per_hour:.2f} open={count_open_positions()} last_action={last_action_age:.0f}s"
                )
                log("INFO", msg.replace("\n", " | "))
                tg_send(cfg, msg)
                stats["last_heartbeat_ms"] = now_ms

            # fetch candles + ticker
            ohlcv = fetch_with_retry(
                lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=candles_limit),
                log,
                attempts=max_retries,
                delay_seconds=retry_delay,
            )
            if not ohlcv or len(ohlcv) < max(ma_fast, ma_mid, ma_slow) + 5:
                log("WARN", "Waiting for enough candles to compute moving averages...")
                time.sleep(poll_seconds)
                continue

            last_closed = ohlcv[-2]
            prev_closed = ohlcv[-3]
            last_closed_ts = int(last_closed[0])

            if last_closed_ts == int(state.get("last_processed_candle_ts", 0)):
                time.sleep(poll_seconds)
                continue

            closes = [float(c[4]) for c in ohlcv[:-1]]
            ma_f = ema(closes, ma_fast)
            ma_m = ema(closes, ma_mid)
            ma_s = ema(closes, ma_slow)
            ma_f_prev = ema(closes[:-1], ma_fast)
            atr_pct = atr_percent(ohlcv, volatility_atr_period)
            if ma_f is None or ma_m is None or ma_s is None:
                state["last_processed_candle_ts"] = last_closed_ts
                save_state(state_file, state)
                time.sleep(poll_seconds)
                continue

            last_close = float(last_closed[4])
            prev_close = float(prev_closed[4])

            ticker = fetch_with_retry(lambda: exchange.fetch_ticker(symbol), log, attempts=max_retries, delay_seconds=retry_delay)
            last_price = float(ticker["last"])

            # persist candle cursor early
            state["last_processed_candle_ts"] = last_closed_ts
            save_state(state_file, state)

            # order throttle
            if (now_ms - int(state.get("last_order_ts_ms", 0))) < min_seconds_between_orders * 1000:
                time.sleep(poll_seconds)
                continue

            free_base, free_quote = fetch_with_retry(
                lambda: fetch_free_balances(exchange, base, quote),
                log,
                attempts=max_retries,
                delay_seconds=retry_delay,
            )

            # ---- manage exits for each open position ----
            for pos in state["positions"]:
                if not pos.get("open"):
                    continue

                entry = float(pos["entry_price"])
                amt = float(pos["base_amount"])
                highest = float(pos.get("highest_price", entry))
                slot_id = int(pos.get("slot_id", 0))
                opened_ms = int(pos.get("opened_time_ms", now_ms))
                age_min = (now_ms - opened_ms) / (1000 * 60)

                if last_price > highest:
                    highest = last_price
                    pos["highest_price"] = highest

                if last_price >= entry * (1.0 + breakeven_lock_pct):
                    pos["breakeven_armed"] = True

                stop_price = entry * (1.0 - stop_loss_pct)
                tp_price = entry * (1.0 + take_profit_pct)

                trailing_stop_price = None
                if trailing_enabled and highest >= entry * (1.0 + trailing_activate_pct):
                    trailing_stop_price = highest * (1.0 - trailing_distance_pct)
                    if trailing_stop_price < stop_price:
                        trailing_stop_price = stop_price

                # net floor (anti-fee)
                min_profitable_price = entry * (1.0 + min_net_profit_pct + fee_rate * 2)
                if pos.get("breakeven_armed") and last_price <= entry:
                    min_profitable_price = entry

                exit_reason = None
                allow_sell = True

                if time_exit_enabled and max_hold_minutes > 0 and age_min >= max_hold_minutes:
                    if last_price >= min_profitable_price:
                        exit_reason = f"TIME exit ({age_min:.1f}m)"
                    else:
                        if pos.get("breakeven_armed") and last_price >= entry:
                            exit_reason = f"TIME BE exit ({age_min:.1f}m)"

                if exit_reason is None:
                    if last_price <= stop_price:
                        exit_reason = f"STOP hit ({stop_price:.6f})"
                    elif trailing_stop_price is not None and last_price <= trailing_stop_price:
                        if last_price >= min_profitable_price:
                            exit_reason = f"TRAIL hit ({trailing_stop_price:.6f})"
                        else:
                            allow_sell = False
                            exit_reason = f"TRAIL touched below net floor ({min_profitable_price:.6f})"
                    elif (not trailing_enabled) and last_price >= tp_price:
                        if last_price >= min_profitable_price:
                            exit_reason = f"TP hit ({tp_price:.6f})"
                        else:
                            allow_sell = False
                            exit_reason = f"TP touched below net floor ({min_profitable_price:.6f})"

                est_pnl = estimate_net_pnl_quote(entry, last_price, amt, fee_rate)
                est_pnl_pct = (est_pnl / max(1e-12, entry * amt)) * 100.0
                log(
                    "INFO",
                    (
                        f"POSITION S{slot_id} | price={last_price:.6f} entry={entry:.6f} high={highest:.6f} "
                        f"stop={stop_price:.6f} netFloor={min_profitable_price:.6f} tp={tp_price:.6f} "
                        f"trailStop={(trailing_stop_price if trailing_stop_price else 'n/a')} age={age_min:.1f}m "
                        f"estPnL≈{est_pnl:.4f} {quote} ({est_pnl_pct:+.2f}%)"
                    ),
                )

                if exit_reason and allow_sell:
                    if (now_ms - int(state.get("last_order_ts_ms", 0))) < min_seconds_between_orders * 1000:
                        continue

                    state["trade_seq"] = int(state.get("trade_seq", 0)) + 1
                    instance_id = str((cfg.get("misc") or {}).get("instance_id") or (cfg.get("misc") or {}).get("bot_name") or symbol)
                    client_id = deterministic_client_id(instance_id, symbol, "SELL", last_closed_ts, int(state["trade_seq"]), slot_id)

                    log("SELL", f"{symbol} S{slot_id} | Exit: {exit_reason} | qty={amt}")
                    tg_send(cfg, f"🔴 SELL {symbol} (S{slot_id})\nReason: {exit_reason}\nPrice ~{last_price}")

                    free_base_now, _ = fetch_free_balances(exchange, base, quote)
                    sell_amt_raw = min(amt, free_base_now) if free_base_now > 0 else amt
                    sell_amt = amount_to_precision(exchange, symbol, sell_amt_raw)
                    if sell_amt <= 0:
                        log("WARN", f"S{slot_id} sell amount computed 0; skipping.")
                        continue

                    order, exec_label = execute_order(
                        exchange=exchange,
                        symbol=symbol,
                        side="sell",
                        amount=sell_amt,
                        client_id=client_id,
                        execution_cfg=execution_cfg,
                        dry_run=dry_run,
                        log_fn=log,
                    )

                    filled_exit = safe_float(order.get("filled"), sell_amt) or sell_amt
                    avg_exit = safe_float(order.get("average"), last_price) or last_price

                    fee_paid, fee_ccy = collect_fee_from_order_or_trades(exchange, symbol, order, since_ms=opened_ms - 60_000)
                    pnl_est = estimate_net_pnl_quote(entry, avg_exit, filled_exit, fee_rate)
                    if fee_paid and fee_ccy == quote:
                        gross = (avg_exit - entry) * filled_exit
                        entry_fee_paid = float(pos.get("entry_fee_paid", 0.0) if pos.get("entry_fee_currency", "") == quote else 0.0)
                        pnl_est = gross - fee_paid - entry_fee_paid

                    pnl_pct_est = (pnl_est / max(1e-12, entry * filled_exit)) * 100.0

                    state["last_order_ts_ms"] = now_ms
                    state["realized_profit_quote_net"] = float(state.get("realized_profit_quote_net", 0.0)) + float(pnl_est)

                    if pnl_est < 0:
                        state["consecutive_losses"] = int(state.get("consecutive_losses", 0)) + 1
                        if loss_cd_enabled and max_consecutive_losses > 0 and state["consecutive_losses"] >= max_consecutive_losses:
                            # cap cooldown to 60s
                            cooldown_ms = 60_000
                            state["cooldown_until_ms"] = max(int(state.get("cooldown_until_ms", 0)), now_ms + cooldown_ms)
                            state["consecutive_losses"] = 0
                            log("WARN", "PAUSED after losses for 60s (cap)")
                    else:
                        state["consecutive_losses"] = 0

                    if pnl_est >= 0:
                        log("WIN", f"{symbol} S{slot_id} | Filled @ {avg_exit:.6f} | PnL≈+{pnl_est:.4f} {quote} (+{pnl_pct_est:.2f}%) | {exit_reason} | exec={exec_label}")
                        tg_send(cfg, f"✅ WIN {symbol} (S{slot_id})\nPnL ~+{pnl_est:.4f} {quote} (+{pnl_pct_est:.2f}%)\nExec: {exec_label}")
                        stats["wins"] += 1
                    else:
                        log("LOSS", f"{symbol} S{slot_id} | Filled @ {avg_exit:.6f} | PnL≈{pnl_est:.4f} {quote} ({pnl_pct_est:.2f}%) | {exit_reason} | exec={exec_label}")
                        tg_send(cfg, f"⚠️ LOSS {symbol} (S{slot_id})\nPnL ~{pnl_est:.4f} {quote} ({pnl_pct_est:.2f}%)\nExec: {exec_label}")
                        stats["losses"] += 1

                    stats["trades"] += 1
                    stats["pnl_quote"] += float(pnl_est)
                    stats["last_action_ts_ms"] = now_ms

                    duration_s = (now_ms - opened_ms) / 1000.0
                    journal_row(
                        journal_file,
                        [
                            ts_utc(),
                            symbol,
                            "SELL",
                            "EXIT",
                            exit_reason,
                            slot_id,
                            filled_exit,
                            avg_exit,
                            float(filled_exit) * float(avg_exit),
                            fee_paid,
                            fee_ccy,
                            pnl_est,
                            pnl_pct_est,
                            duration_s,
                            client_id,
                            last_closed_ts,
                            exec_label,
                        ],
                    )

                    slot_id_keep = slot_id
                    pos.update(_default_position(slot_id_keep))
                    pos["slot_id"] = slot_id_keep

                    cd = max(0, cooldown_seconds_after_close)
                    if cd > 0:
                        state["cooldown_until_ms"] = max(int(state.get("cooldown_until_ms", 0)), now_ms + min(60_000, cd * 1000))

                    save_state(state_file, state)

            # ---- entries ----
            open_positions = count_open_positions()
            if open_positions < max_positions:
                if (now_ms - int(state.get("last_entry_time_ms", 0))) < min_seconds_between_entries * 1000:
                    time.sleep(poll_seconds)
                    continue

                bid, ask = get_best_bid_ask(exchange, symbol)
                spr_bps = None
                if bid and ask:
                    spr_bps = spread_bps(bid, ask)
                    if spread_enabled and spr_bps > max_spread_bps:
                        log("INFO", f"SPREAD BLOCK | spread={spr_bps:.1f} bps > max={max_spread_bps:.1f} bps")
                        time.sleep(poll_seconds)
                        continue

                slope_bps = 0.0
                slope_ok = False
                if ma_f_prev and ma_f_prev > 0:
                    slope_bps = ((ma_f - ma_f_prev) / ma_f_prev) * 10000.0
                    slope_ok = slope_bps >= min_fast_slope_bps and ma_f > ma_f_prev

                trend_ok = (ma_f > ma_m) and (ma_m >= ma_s) and slope_ok
                bounce_ok = ((prev_close <= ma_f and last_close > ma_f) or (prev_close <= ma_m and last_close > ma_m))
                if allow_range_entries:
                    bounce_ok = bounce_ok or (prev_close <= ma_f and last_close > ma_f)

                price_ok = (last_close > ma_m) if require_above_ma_mid else True
                dist_above_m = (last_close - ma_m) / ma_m if ma_m else 0.0
                not_overextended = dist_above_m <= max_entry_distance_ma_mid_pct

                range_entry = allow_range_entries and (not trend_ok) and bounce_ok and not_overextended
                if range_entry and (now_ms - last_range_entry_ms) < range_reentry_cooldown_seconds * 1000:
                    range_entry = False

                enter = (trend_ok and bounce_ok and price_ok and not_overextended) or range_entry

                if enter:
                    if atr_pct is None:
                        enter = False
                        log("INFO", "VOL BLOCK | ATR unavailable")
                    elif atr_pct < min_atr_percent:
                        enter = False
                        log("INFO", f"VOL BLOCK | ATR {atr_pct:.3f}% < min {min_atr_percent:.3f}%")
                    elif atr_pct > max_atr_percent:
                        enter = False
                        log("INFO", f"VOL BLOCK | ATR {atr_pct:.3f}% > max {max_atr_percent:.3f}%")

                spread_buffer_pct = 0.0
                if spr_bps is not None:
                    spread_buffer_pct = (spr_bps / 10000.0) * 0.5  # half-spread buffer

                if enter and viability_enabled:
                    ok = profit_viable(
                        entry_price=last_close,
                        take_profit_pct=take_profit_pct + viability_margin_pct,
                        fee_rate=fee_rate,
                        slippage_buffer_pct=slippage_buffer_pct,
                        spread_buffer_pct=spread_buffer_pct,
                        min_net_profit_pct=min_net_profit_pct,
                    )
                    if not ok:
                        enter = False
                        log(
                            "INFO",
                            (
                                "VIABILITY BLOCK | expected net profit below minimum "
                                f"(TP target={(take_profit_pct + viability_margin_pct)*100:.2f}% minNet={min_net_profit_pct*100:.2f}% "
                                f"spreadBuf={spread_buffer_pct*100:.3f}% slipBuf={slippage_buffer_pct*100:.3f}% fee={fee_rate*100:.3f}%)"
                            ),
                        )

                log(
                    "INFO",
                    (
                        f"CANDLE CLOSED | close={last_close:.6f} price={last_price:.6f} | "
                        f"EMA(f/m/s)={ma_f:.6f}/{ma_m:.6f}/{ma_s:.6f} slope={slope_bps:.2f}bps | "
                        f"trend={trend_ok} bounce={bounce_ok} overext_ok={not_overextended} range_entry={range_entry} enter={enter}"
                        + (f" | spread={spr_bps:.1f}bps" if spr_bps is not None else "")
                        + (f" | ATR={atr_pct:.3f}%" if atr_pct is not None else "")
                    ),
                )

                if enter:
                    slot = None
                    for p in state["positions"]:
                        if not p.get("open"):
                            slot = p
                            break
                    if slot is None:
                        time.sleep(poll_seconds)
                        continue

                    if reserve_enabled and free_quote <= min_quote_reserve:
                        log("WARN", f"Reserve guard: free {free_quote:.2f}{quote} <= reserve {min_quote_reserve:.2f}{quote}")
                        time.sleep(poll_seconds)
                        continue

                    free_quote_for_entry = free_quote - (min_quote_reserve if reserve_enabled else 0.0)

                    if free_quote_for_entry < trade_quote_amount:
                        log("WARN", f"Insufficient free quote {free_quote_for_entry:.2f}{quote} for target {trade_quote_amount:.2f}{quote}")
                        time.sleep(poll_seconds)
                        continue

                    entry_quote_cap = compute_dynamic_entry_quote(cfg, free_quote_for_entry if not dry_run else max(free_quote_for_entry, 1.0), open_positions)
                    if not dry_run:
                        entry_quote_cap = min(entry_quote_cap, free_quote_for_entry)

                    if entry_quote_cap <= 0:
                        time.sleep(poll_seconds)
                        continue
                    if min_cost is not None and entry_quote_cap < min_cost:
                        log("WARN", f"Entry cap {entry_quote_cap:.2f} {quote} below min notional {min_cost}.")
                        time.sleep(poll_seconds)
                        continue

                    base_amount_raw = entry_quote_cap / last_price
                    base_amount = amount_to_precision(exchange, symbol, base_amount_raw)
                    if base_amount <= 0:
                        log("WARN", "Computed base amount is 0 after precision. Skipping entry.")
                        time.sleep(poll_seconds)
                        continue
                    if min_amount is not None and base_amount < min_amount:
                        log("WARN", f"Amount {base_amount} < min amount {min_amount}.")
                        time.sleep(poll_seconds)
                        continue

                    if viability_enabled:
                        viable_now = profit_viable(
                            entry_price=last_price,
                            take_profit_pct=take_profit_pct + viability_margin_pct,
                            fee_rate=fee_rate,
                            slippage_buffer_pct=slippage_buffer_pct,
                            spread_buffer_pct=spread_buffer_pct,
                            min_net_profit_pct=min_net_profit_pct,
                        )
                        if not viable_now:
                            log(
                                "INFO",
                                (
                                    "VIABILITY BLOCK (pre-order) | expected net profit below minimum "
                                    f"(TP target={(take_profit_pct + viability_margin_pct)*100:.2f}% "
                                    f"minNet={min_net_profit_pct*100:.2f}% spreadBuf={spread_buffer_pct*100:.3f}% "
                                    f"slipBuf={slippage_buffer_pct*100:.3f}% fee={fee_rate*100:.3f}%)"
                                ),
                            )
                            time.sleep(poll_seconds)
                            continue

                    state["trade_seq"] = int(state.get("trade_seq", 0)) + 1
                    slot_id = int(slot.get("slot_id", 0))
                    instance_id = str((cfg.get("misc") or {}).get("instance_id") or (cfg.get("misc") or {}).get("bot_name") or symbol)
                    client_id = deterministic_client_id(instance_id, symbol, "BUY", last_closed_ts, int(state["trade_seq"]), slot_id)
                    reason = "trend+bounce" if not range_entry else "range-bounce"

                    log("BUY", f"{symbol} S{slot_id} | px={last_price:.6f} qty={base_amount} notional≈{entry_quote_cap:.2f} {quote} | reason={reason}")
                    tg_send(cfg, f"🟢 BUY {symbol} (S{slot_id})\nReason: {reason}\nQty ~{base_amount}\nPrice ~{last_price}")

                    order, exec_label = execute_order(
                        exchange=exchange,
                        symbol=symbol,
                        side="buy",
                        amount=base_amount,
                        client_id=client_id,
                        execution_cfg=execution_cfg,
                        dry_run=dry_run,
                        log_fn=log,
                    )

                    filled = safe_float(order.get("filled"), 0.0) or 0.0
                    avg_price = safe_float(order.get("average"), None)

                    if filled <= 0:
                        log("WARN", f"S{slot_id} entry filled 0 (exec={exec_label}). Skipping slot open.")
                        time.sleep(poll_seconds)
                        continue

                    if avg_price is None:
                        avg_price = last_price

                    notional = filled * float(avg_price)
                    fee_paid, fee_ccy = collect_fee_from_order_or_trades(exchange, symbol, order, since_ms=now_ms - 60_000)

                    slot.update(
                        {
                            "open": True,
                            "entry_price": float(avg_price),
                            "base_amount": float(filled),
                            "highest_price": float(avg_price),
                            "breakeven_armed": False,
                            "adds_count": 0,
                            "entry_quote_spent_est": float(notional),
                            "opened_time_ms": now_ms,
                            "last_action_ms": now_ms,
                            "last_client_id": client_id,
                            "last_execution": exec_label,
                            "entry_fee_paid": float(fee_paid),
                            "entry_fee_currency": str(fee_ccy),
                        }
                    )

                    state["last_entry_time_ms"] = now_ms
                    state["last_order_ts_ms"] = now_ms
                    stats["last_action_ts_ms"] = now_ms
                    if range_entry:
                        last_range_entry_ms = now_ms

                    journal_row(
                        journal_file,
                        [
                            ts_utc(),
                            symbol,
                            "BUY",
                            "ENTRY",
                            reason,
                            slot_id,
                            filled,
                            avg_price,
                            notional,
                            fee_paid,
                            fee_ccy,
                            "",
                            "",
                            "",
                            client_id,
                            last_closed_ts,
                            exec_label,
                        ],
                    )

                    save_state(state_file, state)

            time.sleep(poll_seconds)

        except ccxt.NetworkError as e:
            log("WARN", f"Network error: {e}")
            time.sleep(3)
        except ccxt.ExchangeError as e:
            log("WARN", f"Exchange error: {e}")
            time.sleep(3)
        except KeyboardInterrupt:
            log("WARN", "Stopping bot (Ctrl+C).")
            tg_send(cfg, "🛑 BOTV12 stopped by user.")
            sys.exit(0)
        except Exception as e:
            log("ERROR", f"Unexpected error: {repr(e)}")
            tg_send(cfg, f"❌ BOTV12 error: {repr(e)}")
            time.sleep(3)


if __name__ == "__main__":
    main()
