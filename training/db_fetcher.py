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

  
