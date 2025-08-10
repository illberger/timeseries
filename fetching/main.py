import pyodbc
from binance.client import Client
from binance.enums import HistoricalKlinesType
import datetime
"""
Here using the community wrapper python-binance
'pip install python-binance'

"""
max_days = 300  # 2024-03-06 -> 2025-01-01 för stateful LSTM tränad på BTC-kurser
base_start_unix = 1709679600000  # 2024-03-06 (BTC sentiment för 2024 saknas innan detta datum)  # tidigare april 01 2025 = 1743458400000 för stateless LSTM med 1.5m rader

def get_db_connection():

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=BinanceDB;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)


def ms_to_datetime_str(ms):

    dt = datetime.datetime.utcfromtimestamp(ms / 1000.0)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def fetch_candlesticks_from_binance(client, symbol, start_ms, end_ms):

    start_str = ms_to_datetime_str(start_ms)
    end_str = ms_to_datetime_str(end_ms)

    klines = client.get_historical_klines(
        symbol=symbol.upper(),
        interval=Client.KLINE_INTERVAL_5MINUTE,
        start_str=start_str,
        end_str=end_str,
        klines_type=HistoricalKlinesType.SPOT
    )

    # Each kline is:
    # [
    #   1499040000000,      // Open time
    #   "0.01634790",       // Open
    #   "0.80000000",       // High
    #   "0.01575800",       // Low
    #   "0.01577100",       // Close
    #   "148976.11427815",  // Volume
    #   1499644799999,      // Close time
    #   "2434.19055334",    // Quote asset volume
    #   308,                // Number of trades
    #   "1756.87402397",    // Taker buy base asset volume
    #   "28.46694368",      // Taker buy quote asset volume
    #   "17928899.62484339" // Ignore
    # ]

    return klines


def insert_candlesticks_into_db(symbol, interval, klines):
    conn = get_db_connection()
    cursor = conn.cursor()

    for k in klines:
        open_time = k[0]
        open_price = float(k[1])
        high_price = float(k[2])
        low_price = float(k[3])
        close_price = float(k[4])
        volume = float(k[5])
        close_time = k[6]

        cursor.execute("""
            INSERT INTO dbo.CandleSticksBtc
            (Symbol, Interval, OpenTime, OpenPrice, HighPrice, LowPrice, ClosePrice, Volume, CloseTime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, interval, open_time, open_price, high_price, low_price, close_price, volume, close_time
        ))
    conn.commit()
    cursor.close()
    conn.close()


def fetch_all_symbols(client):
    tickers = client.get_symbol_ticker()
    symbols = [ticker['symbol'] for ticker in tickers]
    return symbols


def main():

    """
    inga api-nycklar krävs.
    :return:
    """
    client = Client(api_key="",
                    api_secret="")

    symbols = fetch_all_symbols(client)


    day_ms = 86400000
    #import random

    days = max_days  # 30
    test1 = (days * 288)
    test2 = (days * 288) + 1
    end_time = base_start_unix + days * day_ms


    """
    main loop för 5-minutersdatan
    """
    for idx, symbol in enumerate(symbols, 1):
        if not symbol.startswith("BTC"):
            continue
        try:
            klines = fetch_candlesticks_from_binance(client, symbol, base_start_unix, end_time)
            if klines:
                if len(klines) == test1 or len(klines) == test2:
                    insert_candlesticks_into_db(symbol, "5m", klines)
                    print(f"[{idx}/{len(symbols)}] Inserted {len(klines)} klines for {symbol}")
                else:
                    print(f"Incomplete data for {symbol}, only {len(klines)} klines when we need {test1}")
            else:
                print(f"[{idx}/{len(symbols)}] No data for {symbol} in this period.")
        except Exception as e:
            print(f"[{idx}/{len(symbols)}] Error fetching klines for {symbol}: {e}")

    print("Completed inserting expanded dataset!")

if __name__ == "__main__":
    main()
