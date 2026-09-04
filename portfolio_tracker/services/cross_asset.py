"""Bounded non-Streamlit peer cache shared by the UI and headless collector."""
from concurrent.futures import Future, TimeoutError
from threading import Lock, Thread
from time import monotonic, time

from ..analytics.cross_correlation import PEERS, apply_cross_context, build_cross_context, unavailable

_LOCK = Lock()
_CACHE = {}  # At most the two explicitly supported peers; one daemon worker each.
TTL = 300


def _download_peer(symbol):
    import yfinance as yf
    from ..config import DATA_DIR
    from .quant_market_data import _normalize_frame
    cache = DATA_DIR / "yfinance_cache"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    # Ticker.history avoids yf.download's shared multi-ticker result dictionary.
    ticker = yf.Ticker(symbol)
    common = dict(auto_adjust=False, prepost=False, timeout=5, raise_errors=True)
    daily = _normalize_frame(ticker.history(period="1y", interval="1d", **common), symbol)
    intraday = _normalize_frame(ticker.history(period="5d", interval="5m", **common), symbol)
    return daily, intraday


def prefetch_cross_asset(symbol):
    peer = PEERS.get(symbol)
    if not peer:
        return None
    with _LOCK:
        entry = _CACHE.get(peer)
        if entry is not None:
            future, created, bucket = entry
            if not future.done() or (monotonic() - created < 60 if future.exception() else
                                     monotonic() - created < TTL and bucket == int(time() // 300)):
                return future
        future = Future()
        _CACHE[peer] = (future, monotonic(), int(time() // 300))
        def fetch():
            try:
                future.set_result(_download_peer(peer))
            except Exception as exc:
                future.set_exception(exc)
        Thread(target=fetch, name=f"cross-{peer}", daemon=True).start()
        return future


def enrich_cross_asset(analysis, intraday, daily, *, now=None, timeout=2.0, peer_loader=None):
    """Download errors/timeout never block the primary analysis or write account data."""
    symbol = analysis.symbol
    if symbol not in PEERS:
        return analysis
    try:
        if intraday is None or daily is None or intraday.empty or daily.empty:
            raise ValueError("Faltan velas del activo principal.")
        peer_daily, peer_intraday = (peer_loader(PEERS[symbol]) if peer_loader else
                                     prefetch_cross_asset(symbol).result(timeout=max(0, timeout)))
        context = build_cross_context(symbol, daily, peer_daily, intraday, peer_intraday, as_of=now)
    except TimeoutError:
        context = unavailable(symbol, "descarga pendiente; se reintentará en la próxima actualización")
    except Exception as exc:
        context = unavailable(symbol, str(exc))
    return apply_cross_context(analysis, context)
