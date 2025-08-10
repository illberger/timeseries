# db_fetcher.py
import pandas as pd
import pyodbc
import tensorflow as tf
import numpy as np
from line_profiler import profile

symbol = "BTCUSDT"  # Testing for finding good architecture
feature_n = 8

class DBFetcher:
    """
    Fetches training data \n
    This class has huge overhead when streaming with tensorflow at every step. This can be used to
    cache everything at first epoch, making subsequent epochs go faster
    """
    @profile
    def __init__(self, server='localhost', database='BinanceDB'):
        """
        you would need to set up your own schema if you want to train model using this project
        :param server:
        :param database:
        """

        """
        dbo.CandleSticksBtc   count(*) == 1123213
        """

        self.conn_str = (
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

        self.acquery = """
            SELECT
              cs.OpenTime,
              cs.ClosePrice
            FROM dbo.CandleSticksBtc_n48 AS cs
            WHERE cs.Symbol = ?
            ORDER BY cs.OpenTime
        """


        self.tquery = """
                    SELECT 
                        cs.OpenTime,
                        cs.OpenPrice, cs.HighPrice, cs.LowPrice, cs.ClosePrice, cs.Volume,
                        COALESCE(pm.SentimentMean, 0.0) AS Sentiment
                    FROM dbo.CandleSticksBtc AS cs
                    LEFT JOIN dbo.Polymarket AS pm
                      ON pm.PriceTimeMs = cs.OpenTime
                    ORDER BY cs.Symbol, cs.OpenTime;
                """

        # Denna har används för btcusdt_optmized och btc_usdt_sentz
        self.tequery = f"""
                    SELECT
                      cs.OpenTime,
                      cs.OpenPrice, cs.HighPrice, cs.LowPrice, cs.ClosePrice, cs.Volume,
                      COALESCE(pm.SentimentMean, 0.0) AS Sentiment
                    FROM dbo.CandleSticksBtc AS cs
                    LEFT JOIN dbo.Polymarket AS pm
                      ON pm.PriceTimeMs = cs.OpenTime
                    WHERE cs.Symbol = '{symbol}'
                    ORDER BY cs.OpenTime;
                """

        # 2025-07-24 testar utan sentiment
        self.testquery = f"""
                            SELECT
                              cs.OpenTime,
                              cs.OpenPrice, cs.HighPrice, cs.LowPrice, cs.ClosePrice, cs.Volume
                            FROM dbo.CandleSticksBtc AS cs
                            WHERE cs.Symbol = '{symbol}'
                            ORDER BY cs.OpenTime;
                        """
        self.n48z_query = """
                    SELECT CandleStickId, OpenTime
                    FROM dbo.CandleSticksBtc_n48z
                    ORDER BY CandleStickId
                """

        self.query = f"""
                    SELECT
                      cs.OpenPriceZ,
                      cs.HighPriceZ,
                      cs.LowPriceZ,
                      cs.ClosePriceZ,
                      cs.VolumeZ,
                      cs.SinTZ,
                      cs.CosTZ,
                      COALESCE(pm.SentimentMean, 0.0) AS Sentiment
                    FROM dbo.CandleSticksBtc_n48z AS cs
                    LEFT JOIN dbo.Polymarket_n48  AS pm
                      ON pm.PriceTimeMs = cs.OpenTime
                    WHERE cs.Symbol = '{symbol}'
                    ORDER BY cs.OpenTime;
                """
        """
        This is the query used (query)
        """
        self.nonsentquery = f"""
                            SELECT
                              cs.OpenPriceZ,
                              cs.HighPriceZ,
                              cs.LowPriceZ,
                              cs.ClosePriceZ,
                              cs.VolumeZ,
                              cs.SinTZ,
                              cs.CosTZ
                            FROM dbo.CandleSticksBtc_n48z AS cs
                            WHERE cs.Symbol = '{symbol}'
                            ORDER BY cs.OpenTime;
                        """

    @profile
    def row_count(self) -> int:
        """
        Räknar antalet rader i databasen
        :return:
        """
        """
        OBS! Dubbelkolla namnet på tabellen!
        """
        q = "SELECT COUNT(*) FROM dbo.CandleSticksBtc_n48z"
        with pyodbc.connect(self.conn_str) as conn, conn.cursor() as cur:
            cur.execute(q)
            return cur.fetchone()[0]

    """
    @profile
    def _row_generator(self):
        conn = pyodbc.connect(self.conn_str)
        cur = conn.cursor()
        #cur.arraysize = 1024  # (13 * 86401) rows in dbo.CandleSticksBtw + 86401 rows in dbo.Polymarket
        cur.execute(self.query)
        for row in cur:
            yield (
                #row.Symbol,
                row.OpenTime,
                float(row.OpenPrice),
                float(row.HighPrice),
                float(row.LowPrice),
                float(row.ClosePrice),
                float(row.Volume),
                float(row.Sentiment),  # Sentiment
            )
        conn.close()
    """

    def _row_generator(self):
        with pyodbc.connect(self.conn_str) as conn:
            cur = conn.cursor()
            cur.execute(self.query)
            for row in cur:
                yield np.asarray(row, dtype=np.float32)

    def get_dataset(self) -> tf.data.Dataset:
        output_signature = tf.TensorSpec(shape=(feature_n,), dtype=tf.float32)
        return tf.data.Dataset.from_generator(
            self._row_generator,
            output_signature=output_signature
        )

    def fetch_close_series(self, symbol: str) -> pd.DataFrame:
        """
        Fetches the chronological ClosePrice series for a given symbol using pyodbc cursor.

        :param symbol: Trading symbol, e.g. 'BTCUSDT'
        :return: DataFrame with columns ['OpenTime', 'ClosePrice'], ordered by OpenTime
        """
        conn = pyodbc.connect(self.conn_str)
        try:
            cursor = conn.cursor()
            cursor.execute(self.acquery, symbol)
            cols = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
        finally:
            conn.close()

        df = pd.DataFrame.from_records(rows, columns=cols)
        df['OpenTime'] = pd.to_datetime(df['OpenTime'])
        return df

    def compute_autocorrelation(self, series: pd.Series, nlags: int = 2000) -> pd.Series:
        """
        Computes the autocorrelation function up to a given number of lags.

        :param series: Pandas Series of numeric values (e.g. ClosePrice)
        :param nlags: Number of lags to compute (default 2000)
        :return: Pandas Series indexed by lag (0..nlags) with autocorrelation values
        """
        x = series.values - series.values.mean()
        var = series.values.var()
        autocorr = []
        for t in range(nlags + 1):
            if t == 0:
                autocorr.append(1.0)
            else:
                cov = (x[:-t] * x[t:]).mean()
                autocorr.append(cov / var)
        return pd.Series(autocorr, index=range(nlags + 1))

    """
    # Exempel för pyodbc + pandas update
    def fetch_n48z_table(self) -> pd.DataFrame:
        conn = pyodbc.connect(self.conn_str)
        try:
            cursor = conn.cursor()
            cursor.execute(self.n48z_query)
            cols = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
        finally:
            conn.close()
        df = pd.DataFrame.from_records(rows, columns=cols)
        df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
        return df

    def update_time_features(self, df: pd.DataFrame):
        minutes = df['OpenTime'].dt.hour * 60 + df['OpenTime'].dt.minute
        angles = 2 * np.pi * minutes / (24 * 60)
        sin_raw = np.sin(angles)
        cos_raw = np.cos(angles)
        sin_z = (sin_raw - sin_raw.mean()) / sin_raw.std(ddof=0)
        cos_z = (cos_raw - cos_raw.mean()) / cos_raw.std(ddof=0)
        df_loc = df.copy()
        df_loc['SinTZ'] = sin_z
        df_loc['CosTZ'] = cos_z
        params = list(zip(df_loc['SinTZ'].astype(float),
                          df_loc['CosTZ'].astype(float),
                          df_loc['CandleStickId'].astype(int)))
        update_sql = (
            "UPDATE dbo.CandleSticksBtc_n48z "
            "SET SinTZ = ?, CosTZ = ? "
            "WHERE CandleStickId = ?"
        )
        conn = pyodbc.connect(self.conn_str)
        try:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            cursor.executemany(update_sql, params)
            conn.commit()
        finally:
            conn.close()
    """
    """
    @profile
    def get_dataset(self) -> tf.data.Dataset:
        output_signature = (
            #tf.TensorSpec(shape=(), dtype=tf.string),   # Symbol
            tf.TensorSpec(shape=(), dtype=tf.int64),    # OpenTime
            tf.TensorSpec(shape=(), dtype=tf.float32),  # OpenPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # HighPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # LowPrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # ClosePrice
            tf.TensorSpec(shape=(), dtype=tf.float32),  # Volume
            tf.TensorSpec(shape=(), dtype=tf.float32),  # Sentiment
        )

        return tf.data.Dataset.from_generator(
            self._row_generator,
            output_signature=output_signature,
        )
    """
