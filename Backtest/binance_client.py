# /Backtest/binance_client.py
# NumPy v1.23.5
"""
Requires running 'pip install binance-connector'
"""
import logging
from collections import defaultdict
from typing import Optional, final

from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.websocket.spot.websocket_api import SpotWebsocketAPIClient
import json
import time
from math import sin, cos, pi
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx
import numpy as np
import datetime as dt
from CONSTANTS import SEQ_LEN, WINDOW_LENGTH_DAY, FIXED_CLOB_TOKEN_IDS

DOWNSAMPLE = int(WINDOW_LENGTH_DAY / 48)

def fetch_5m_values_for_token(clob_token_id: str, start_s: int, end_s: int) -> list[float]:
    _client = _client = httpx.Client(timeout=10.0)
    url = "https://clob.polymarket.com/prices-history"
    params = {"market": clob_token_id, "startTs": start_s, "endTs": end_s, "fidelity": 5}
    try:
        r = _client.get(url, params=params); r.raise_for_status()
        hist = r.json().get("history", [])
        return [pt["p"] for pt in hist if "p" in pt]
    except:
        return []


class BinanceWebSocketClient:
    """
    A wrapper around the Binance SpotWebsocketStreamClient to subscribe to
    candlestick (kline) data for multiple symbols. Maintains a dictionary
    of the latest candles in memory, keyed by symbol.
    """
    #PARTIAL_MOD = 10

    def __init__(self, is_sent_feature: bool):
        self.is_sent_feature: final = is_sent_feature
        self.current_history_symbol = None
        self.logger = logging.getLogger(__name__)
        self.closed_candles = {}
        self.sentiment_series = []
        self.partial_ctr = defaultdict(int)
        self.running_stats: dict[str, BinanceWebSocketClient.RunningStats] = {}
        self.subscribed_symbols = set()
        self.LIMIT_CANDLES_STORAGE = int(SEQ_LEN * 2)
        self.last_mean = 0.0
        self.hist_client = SpotWebsocketAPIClient(
            on_message=self.hist_msg_handler,
            on_close=self.has_closed()
        )

        self.client = SpotWebsocketStreamClient(
            on_message=self._on_message,
            is_combined=True,
        )

    class RunningStats:
        """
        """

        def __init__(self, n_feats: int):
            self.n = 0
            self.mean = np.zeros(n_feats, dtype=np.float64)
            self.M2 = np.zeros(n_feats, dtype=np.float64)

        def _add(self, x: np.ndarray):
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

        def _remove(self, x: np.ndarray):
            if self.n <= 1:
                self.n = 0
                self.mean.fill(0.0)
                self.M2.fill(0.0)
                return
            delta = x - self.mean
            self.n -= 1
            self.mean -= delta / self.n
            delta2 = x - self.mean
            self.M2 -= delta * delta2
            self.M2 = np.maximum(self.M2, 0.0)

        def update(self, x_new: np.ndarray, x_old: np.ndarray | None = None):
            self._add(x_new)
            if x_old is not None:
                self._remove(x_old)

        @property
        def std(self) -> np.ndarray:
            return np.sqrt(self.M2 / max(self.n, 1)) + 1e-6

    def robust_mean(self, x, q_low=5, q_high=95):
        if len(x) == 0:
            return self.last_mean
        low, high = np.percentile(x, [q_low, q_high])
        x_clipped = np.clip(x, low, high)
        ret = float(np.mean(x_clipped))
        self.last_mean = ret
        return ret

    def is_streaming(self, symbol: str) -> bool:
        return symbol.upper() in self.subscribed_symbols

    def _on_message(self, _, raw: str):
        """
        Lyssnare som bifogar stängda candlesticks som skickas från binances servrar varje 5-minuts-klockslag. WS-meddelanden
        kommer i JSON-format ungefär varje sekund. Se text-block nedan i denna metod gällande detta.
        :param _:
        :param raw:
        :return:
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            self.logger.error("bad json")
            return

        data = msg.get("data", {})
        if data.get("e") != "kline":
            return

        k = data["k"]
        symbol = k["s"]
        closed = k["x"]

        if closed:
            self.closed_candles[symbol].append(k)
            self.closed_candles[symbol] = self.closed_candles[symbol][-self.LIMIT_CANDLES_STORAGE:]
            self.partial_ctr[symbol] = 0
            return
        """
        # Om man vill ta in mer information, exempelvis varje rad (denna WSS-endpoint skickar tickers var 1:e sekund),
        # kan man avkommentera följande kod. Det betyder dock att Sin-, Cos-signalerna inte kommer passas in på det modellen
        # sett under träning. Detta hade kanske kunnat lösas med att överskugga candle_to_row där tid-nämnaren är den nya
        # tidrutan för "partial candles". Alternativt så hade man fått träna modellen på kanske 1-minute-candlesticks 
        # istället. 
        # Partial k-lines every 10th msg
        self.partial_ctr[symbol] += 1
        if self.partial_ctr[symbol] % self.PARTIAL_MOD != 0:
            return

        semi = k.copy() # "k" är hela K-linan med OHLC etc
        semi["t"] = data["E"]  # Replace the openTime before appending. # Realtiden finns innan "k"
        semi["x"] = True  # Ej essential metadata (används inte i träningen).
        self.closed_candles[symbol].append(semi)
        self.closed_candles[symbol] = self.closed_candles[symbol][-200:]
        """

    def fetch_historical_5m_candles(self,
                                    symbol: str,
                                    shift: int,
                                    max_days: int,
                                    start_time_ms: int,
                                    lookback_minutes: int = 1440):
        """
        Ask the historical websocket client to fetch 5m candles for the specified lookback period.
        The returned JSON will be passed to hist_msg_handler.
        """
        symbol = symbol.upper()
        #print(f"Fetching {symbol}")
        if symbol not in self.closed_candles:
            self.closed_candles[symbol] = []
        # Store the symbol as the current symbol for historical data
        self.current_history_symbol = symbol

        day_ms = 24 * 60 * 60 * 1000
        window_minutes = lookback_minutes * 2
        window_ms = window_minutes * 60 * 1000
        now_ms = start_time_ms

        end_time = now_ms - ((max_days - shift - 1) * day_ms)
        start_time = end_time - window_ms

        client = self.hist_client

        client.klines(
            symbol=symbol,
            interval="5m",
            startTime=start_time,
            endTime=end_time,
            limit=int(window_minutes/5)
        )

    def compute_sentiment_series(self,
                                 fetch_shift: int,
                                 max_days: int,
                                 start_date_ms: int,
                                 lookback_minutes: int = 1440) -> list[Optional[float]]:
        """
        Returnerar en lista med 288 sentimentvärden för givna params.
        """

        day_ms = 24*60*60*1000
        lookback_minutes = lookback_minutes * 2
        seq_len = lookback_minutes // 5

        end_global = start_date_ms - ((max_days - fetch_shift - 1)*day_ms)
        start_global = end_global - lookback_minutes*60*1000
        start_s = start_global // 1000
        end_s = end_global // 1000

        full = {}
        for tok in FIXED_CLOB_TOKEN_IDS:
            full[tok] = fetch_5m_values_for_token(tok, start_s, end_s)
            time.sleep(0.1)

        out: list[Optional[float]] = []
        for i in range(seq_len):
            vals = [series[i] for series in full.values() if i < len(series)]
            if vals:
                out.append(self.robust_mean(vals))
            else:
                out.append(None)
        return out

    def hist_msg_handler(self, _, message: str):
        """
        Lyssnare för historiska candlesticks.
        :param _:
        :param message:
        :return:
        """
        #print(message)
        try:
            message = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode historical message: {message}")
            return

        if "result" not in message:
            return

        results = message["result"]
        results = results[::DOWNSAMPLE]
        for arr in results:
            kline_dict = {
                't': arr[0],  # open time
                'o': arr[1],  # open price
                'h': arr[2],  # high price
                'l': arr[3],  # low price
                'c': arr[4],  # close price
                'v': arr[5],  # volume
                'x': True,  # historical candles are closed
                's': message.get("symbol", self.current_history_symbol or "UNKNOWN"),
                'i': "5m"  # interval
            }
            if self.is_sent_feature:
                series = getattr(self, "sentiment_series", None)
                idx = len(self.closed_candles.get(kline_dict['s'], []))
                kline_dict['sentiment'] = series[idx] if series and idx < len(series) else None

            if kline_dict['s'] == "UNKNOWN":
                kline_dict['s'] = self.current_history_symbol or "UNKNOWN"

            symbol_key = kline_dict['s']
            if symbol_key not in self.closed_candles:
                self.closed_candles[symbol_key] = []
            self.closed_candles[symbol_key].append(kline_dict)
            # Anropa compute_sentiment_and_append_to_self.closed_candles(self, symbol_key) (vi vet ju att det gäller "BTCUSDC")

        for sym in self.closed_candles:
            self.closed_candles[sym].sort(key=lambda x: x['t'])
            self.closed_candles[sym] = self.closed_candles[sym][: self.LIMIT_CANDLES_STORAGE]

        #print(f"Got message for {sym}")
        self.logger.info(
            f"Historical data loaded: {{ {', '.join(f'{sym}: {len(self.closed_candles[sym])}' for sym in self.closed_candles)} }}")

    def has_closed(self):
        """
        Är tänkt att vara en callback för när en viss WS-klient har stängts (ej lyssnare). Denna är ej klar.
        :return:
        """
        print("A WSS client has closed.")

    def subscribe_to_klines(self, symbols, interval="5m"):
        """
        Använder realtids-WSS-klienten för att streama candlesticks för en viss symbol.
        :param symbols:
        :param interval:
        :return:
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            self.client.kline(symbol.lower(), interval=interval)
            self.logger.info(f"Subscribed to {symbol} @ {interval} klines.")
            self.subscribed_symbols.add(symbol.upper())

    def get_latest_sequence(self, symbol: str, shift: int, seq_len=SEQ_LEN):
        """
        Holds a window of length seq_len * 2. I.e., it returns a Z-scored tensor-slice of a 2 day window.\n
        Seq_len == 24h. Timestep may vary with DOWNSAMPLE.
        :param symbol:
        :param shift:
        :param seq_len:
        :return:
        """

        n_feats = 7
        rows = self.closed_candles.get(symbol.upper(), [])[shift:shift + seq_len]
        if len(rows) < seq_len:
            return None, None, None, None

        stats = self.running_stats.setdefault(
            symbol.upper(),
            BinanceWebSocketClient.RunningStats(n_feats=n_feats)
        )

        full_last_row = self.candle_to_row(rows[-1])
        x_new = np.array(full_last_row[:n_feats], dtype=np.float64)

        x_old = None
        if len(self.closed_candles[symbol]) > self.LIMIT_CANDLES_STORAGE - seq_len:
            old_row = self.closed_candles[symbol][-seq_len - 1]
            x_old = np.array(self.candle_to_row(old_row)[:n_feats], dtype=np.float64)

        stats.update(x_new, x_old)
        x_raw = np.array([self.candle_to_row(r) for r in rows], dtype=np.float32)
        feats = x_raw[:, :n_feats]
        sent = x_raw[:, n_feats:]
        if stats.n < 30:
            mean = feats.mean(axis=0, keepdims=True)
            std = feats.std(axis=0, keepdims=True) + 1e-6
        else:
            mean = stats.mean
            std = stats.std

        feats_scaled = (feats - mean) / std
        feats_scaled = np.concatenate([feats_scaled, sent], axis=1).astype(np.float32)
        last_open_time = rows[-1]['t']
        return feats_scaled, None, x_raw, last_open_time, float(std[0][3]), float(mean[0][3])

    def candle_to_row(self, k):
        minutes = (dt.datetime.utcfromtimestamp(k['t'] / 1000)
                   .hour * 60 +
                   dt.datetime.utcfromtimestamp(k['t'] / 1000).minute)
        sin_t = sin(2 * pi * minutes / (24 * 60))
        cos_t = cos(2 * pi * minutes / (24 * 60))
        return [float(k['o']), float(k['h']), float(k['l']),
                float(k['c']), float(k['v']), sin_t, cos_t,
                float(k.get('sentiment') or 0.0)]

    def stop(self):
        """
        Stops all active streams.
        """
        self.logger.info("Stopping all websocket streams.")
        self.client.stop()
