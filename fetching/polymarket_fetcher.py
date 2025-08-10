# /fetching/polymarket_fetcher.py
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime


import httpx
import pyodbc
from main import base_start_unix, max_days

import logging
logging.basicConfig(level=logging.INFO)
from datetime import datetime, timezone


FIXED_CLOB_TOKEN_IDS = [
    #"112540911653160777059655478391259433595972605218365763034134019729862917878641",       # BTC 2025 (slutpris $1M) (Yes)
    "83894672511259544049673946661753374355328822374216474995072428966535091173758",        # US National BTC reserve 2025 (Yes)
    #"6611523844508119551956870980262427159329487430981844538371350199749910874741",        # Solana all time high by June 30 2025? (Yes)
    #"52696967762983156376661808083218380818225074723063198070857375460800745709299",        # Will a new country buy Bitcoin in 2025? (Yes)
    #"3074539347152748632858978545166555332546941892131779352477699494423276162345",        # MicroStrategy Sells Bitcoin in 2025? (No)
    #"45956246277175804727891136697450869076742282101236359440190169889077896442731",        # BTC up or down in Q2 2025? (Yes)
]

FIXED_CLOB_TOKEN_IDS_2024 = [
    "64903093311385616430821497488306433314807585397286521531639186532059591846310",    # Will BTC hit $100k in 2024? Avbl. from 2024-03-05 till 2024-12-05 (YES)
    "82312610647883846337514556553184069901768182428250018668626625904817742551511",    # Bitcoin New ATH in 2024? Avbl. from 2024-09-21 till 2024-11-05 (YES)
    "109660501215307561524139186012001957478463103025736937371539654796071109353738",   # Will MicroStrategy purchase more BTC in 2024? Avbl. frm 2024-09-21 -> 2024-11-05 (YES)
    "23826820671086031790678277644651229919338596070531413478027516205774692378039",    # US government BTC reserves in 2024? Avbl. from 2024-07-13 -> 2025-01-01 (YES)
]
_client = httpx.Client(timeout=10.0)


def fetch_5m_values_for_token(clob_token_id: str, start_s: int, end_s: int) -> list[float]:
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market":   clob_token_id,
        "startTs":  start_s,
        "endTs":    end_s,
        "fidelity": 5,
    }
    try:
        r = _client.get(url, params=params)
        r.raise_for_status()
        return [pt["p"] for pt in r.json().get("history", []) if "p" in pt]
    except Exception:
        return []

def main():
    conn_str        = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=localhost;"
        "DATABASE=BinanceDB;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    start_date_ms = base_start_unix
    ms_per_day = 24 * 60 * 60 * 1000
    ms_per_5m = 5 * 60 * 1000
    lookback_count = 24 * 60 // 5  # = 288
    total_steps = max_days * lookback_count + 1
    tz_plus2 = timezone(timedelta(hours=2))

    start_ms_global = start_date_ms
    full_series = {tok: [] for tok in FIXED_CLOB_TOKEN_IDS_2024}
    for day in range(max_days):
        day_start_ms = start_ms_global + day * ms_per_day
        day_end_ms = day_start_ms + ms_per_day
        start_s = day_start_ms // 1000
        end_s = day_end_ms // 1000

        with ThreadPoolExecutor(max_workers=len(FIXED_CLOB_TOKEN_IDS_2024)) as ex:
            futures = {
                ex.submit(fetch_5m_values_for_token, tok, start_s, end_s): tok
                for tok in FIXED_CLOB_TOKEN_IDS_2024
            }
            for fut in futures:
                tok = futures[fut]
                vals = fut.result()
                if len(vals) != lookback_count:
                    logging.warning(
                        f"Dag {day+1}/{max_days}, token {tok[:8]}… gav {len(vals)} punkter (förväntat {lookback_count})"
                    )
                full_series[tok].extend(vals)
        if day < 50:
            time.sleep(0.5)
        else:
            time.sleep(0.7)

    cnxn = pyodbc.connect(conn_str)
    cursor = cnxn.cursor()

    for i in range(total_steps):

        windows = [
            full_series[tok][i : i + lookback_count]
            for tok in FIXED_CLOB_TOKEN_IDS_2024
            if len(full_series[tok][i : i + lookback_count]) == lookback_count
        ]

        if windows:
            total_vals = sum(sum(w) for w in windows)
            count_vals = sum(len(w) for w in windows)
            sentiment = total_vals / count_vals
        else:
            sentiment = None

        current_end_ms = start_ms_global + i * ms_per_5m
        price_time = datetime.fromtimestamp(current_end_ms / 1000, tz=tz_plus2)

        cursor.execute("""
            MERGE dbo.Polymarket AS target
            USING (SELECT ? AS PriceTime, ? AS SentimentMean) AS src
              ON target.PriceTime = src.PriceTime
            WHEN MATCHED THEN 
              UPDATE SET SentimentMean = src.SentimentMean
            WHEN NOT MATCHED THEN
              INSERT (PriceTime, SentimentMean) VALUES (src.PriceTime, src.SentimentMean);
        """, price_time, sentiment)

        print(f"[{i+1:4d}/{total_steps}] {price_time.isoformat()} → "
              + (f"{sentiment:.4f}" if sentiment is not None else "NULL"))

    cnxn.commit()
    cursor.close()
    cnxn.close()


if __name__ == "__main__":
    main()
