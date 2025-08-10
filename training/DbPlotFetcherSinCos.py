import pyodbc
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

class DBPlotFetcherSinCos:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=BinanceDB;"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        self.symbol = symbol.upper()

    def get_raw_sincos(self):
        """
        Fetch raw OpenTime and compute sin/cos of time-of-day:
          minutes = (OpenTime/60000) % 1440
          sin_raw = sin(2π * minutes/1440)
          cos_raw = cos(2π * minutes/1440)
        Returns:
            times_raw: np.ndarray shape (N,)
            sin_raw:   np.ndarray shape (N,)
            cos_raw:   np.ndarray shape (N,)
        """
        query = f"""
            SELECT OpenTime
            FROM dbo.CandleSticksBtc_n48
            WHERE Symbol = '{self.symbol}'
            ORDER BY OpenTime
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()

        times_raw = np.array([row.OpenTime for row in rows], dtype=np.int64)
        # compute minutes-of-day for each timestamp
        minutes = ((times_raw // 60000) % 1440).astype(np.float32)
        factor  = 2 * pi / 1440.0
        sin_raw = np.sin(factor * minutes)
        cos_raw = np.cos(factor * minutes)
        return times_raw, sin_raw, cos_raw

    def get_global_sincos_z(self):
        """
        Fetch the precomputed globally Z-scored SinTZ and CosTZ:
        Returns:
            times_z:  np.ndarray shape (M,)
            sin_z:    np.ndarray shape (M,)
            cos_z:    np.ndarray shape (M,)
        """
        query = f"""
            SELECT OpenTime, SinTZ, CosTZ
            FROM dbo.CandleSticksBtc_n48z
            WHERE Symbol = '{self.symbol}'
            ORDER BY OpenTime
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()

        times_z = np.array([row.OpenTime for row in rows], dtype=np.int64)
        sin_z   = np.array([row.SinTZ    for row in rows], dtype=np.float32)
        cos_z   = np.array([row.CosTZ    for row in rows], dtype=np.float32)
        return times_z, sin_z, cos_z
