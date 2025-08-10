import pyodbc
import numpy as np
import matplotlib.pyplot as plt


class DBPlotFetcherSentiment:
    def __init__(self, symbol: str = "BTCUSDT"):
        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=localhost;"
            "DATABASE=BinanceDB;"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )
        self.symbol = symbol.upper()

    def get_global_sentiment_z(self):
        """
        Fetches the globally Z-scored SentimentZ and timestamps.
        Returns:
            times_z: np.ndarray of shape (N,)
            z_sent:  np.ndarray of shape (N,)
        """
        query = f"""
            SELECT PriceTimeMs, SentimentZ
            FROM dbo.Polymarket_Z AS pmz
            ORDER BY PriceTimeMs
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        times_z = np.array([row.PriceTimeMs for row in rows], dtype=np.int64)
        z_sent  = np.array([row.SentimentZ    for row in rows], dtype=np.float32)
        return times_z, z_sent

    def get_raw_sentiment(self):
        """
        Fetches raw SentimentMean and timestamps.
        Returns:
            times_raw: np.ndarray of shape (M,)
            raw_sent:  np.ndarray of shape (M,)
        """
        query = f"""
            SELECT PriceTimeMs, SentimentMean
            FROM dbo.Polymarket AS pm
            ORDER BY PriceTimeMs
        """
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        times_raw = np.array([row.PriceTimeMs    for row in rows], dtype=np.int64)
        raw_sent  = np.array([row.SentimentMean for row in rows], dtype=np.float32)
        return times_raw, raw_sent

