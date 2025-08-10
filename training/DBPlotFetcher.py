import pyodbc
import numpy as np
import matplotlib.pyplot as plt


class DBPlotFetcher:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER=localhost;"
            f"DATABASE=BinanceDB;"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        self.symbol = symbol.upper()

    def get_global_close_z(self):
        """
        Fetches the globally Z-scored ClosePriceZ and timestamps.
        Returns:
            times_global: np.ndarray of shape (N,)
            z_global:    np.ndarray of shape (N,)
        """
        query = f"""
            SELECT OpenTime, ClosePriceZ
            FROM dbo.CandleSticksBtc_Z
            WHERE Symbol = '{self.symbol}'
            ORDER BY OpenTime
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        times_global = np.array([row.OpenTime for row in rows], dtype=np.int64)
        z_global = np.array([row.ClosePriceZ for row in rows], dtype=np.float32)
        return times_global, z_global

    def get_local_close_z(self, sequence_length: int):
        """
        Fetches raw ClosePrice and timestamps, then computes
        per-window Z-score of the closing value at the end of each window.
        Returns:
            times_local: np.ndarray of shape (M,)  where M = N - sequence_length + 1
            z_local:    np.ndarray of shape (M,)
        """
        query = f"""
            SELECT OpenTime, ClosePrice
            FROM dbo.CandleSticksBtc
            WHERE Symbol = '{self.symbol}'
            ORDER BY OpenTime
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        times_raw = np.array([row.OpenTime for row in rows], dtype=np.int64)
        closes_raw = np.array([float(row.ClosePrice) for row in rows], dtype=np.float64)

        N = len(closes_raw)
        M = N - sequence_length + 1
        z_local = np.empty(M, dtype=np.float32)
        times_local = times_raw[sequence_length - 1:]

        # rolling window Z-score of the *last* close in each window
        for i in range(M):
            window = closes_raw[i: i + sequence_length]
            m, s = window.mean(), window.std(ddof=0) + 1e-6
            z_local[i] = (window[-1] - m) / s

        return times_local, z_local
