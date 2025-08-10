# /Backtest/polymarket_fetcher.py
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta, datetime
from typing import Optional

import httpx
from matplotlib import pyplot as plt

# Importera detta om man kör main-metoden i denna fil för logging
#import logging
#logging.basicConfig(level=logging.INFO)

"""
Lista med de fasta clob_token_ids för ett urval av BTC‐relaterade marknader
Notera att dessa kan hämtas ifrån /fetching/polymarket_getter.py.
"cond_id" som behövs kodas in kan hämtas genom att gå in på polymarket.com/event/nagon-sorts-prediktions-marknad

Marknaderna har ett pris för Yes/No för den "fråga" som avgör marknaden. Det är alltså den historiska datan som hämtas.
"""
FIXED_CLOB_TOKEN_IDS = [
    "112540911653160777059655478391259433595972605218365763034134019729862917878641",   # BTC 2025 (slutpris $1M) (Yes)
    "83894672511259544049673946661753374355328822374216474995072428966535091173758",    # US National BTC reserve 2025 (Yes)
    #"6611523844508119551956870980262427159329487430981844538371350199749910874741",     # Solana all time high by June 30 2025? (Yes)
    "52696967762983156376661808083218380818225074723063198070857375460800745709299",    # Will a new country buy Bitcoin in 2025? (Yes)
    #"3074539347152748632858978545166555332546941892131779352477699494423276162345",      # MicroStrategy Sells Bitcoin in 2025? (No)
    "45956246277175804727891136697450869076742282101236359440190169889077896442731",    # BTC up or down in Q2 2025? (Yes)
]
# Andra marknader
#"93592949212798121127213117304912625505836768562433217537850469496310204567695",  # MicroStrategy sells BTC 2025 # Används inte längre då den är pessimistisk (detta är "JA"-priset som är bearish).

_client = httpx.Client(timeout=10.0)


def fetch_5m_values_for_token(
    clob_token_id: str,
    start_time_s: int,
    end_time_s: int
) -> list[float]:
    """
    Fetch 5-minute 'p' values for a polymarket CLOB token.
    Returns empty list on any failure.
    """
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market": clob_token_id,
        "startTs": start_time_s,
        "endTs": end_time_s,
        "fidelity": 5,
    }
    try:
        r = _client.get(url, params=params)
        r.raise_for_status()
        history = r.json().get("history", [])
        if not isinstance(history, list):
            return []
        return [point.get("p") for point in history if "p" in point]
    except Exception:
        return []


def compute_combined_sentiment(
    fetch_shift: int,
    max_days: int,
    start_date_ms: int,
    lookback_minutes: int = 1440
) -> Optional[float]:
    """
    Computes an aggregated sentiment score by fetching 5-minute 'p' values
    for each token in FIXED_CLOB_TOKEN_IDS over the same interval.
    Returns the mean of all collected values, or None if no data.
    """
    #return None # Avkommentera för test 1 för att optimera

    day_ms = 24 * 60 * 60 * 1000
    window_ms = lookback_minutes * 60 * 1000

    end_time_ms = start_date_ms - ((max_days - fetch_shift - 1) * day_ms)
    start_time_ms = end_time_ms - window_ms
    start_s = start_time_ms // 1000
    end_s = end_time_ms // 1000
    expected_count = int(lookback_minutes / 5)

    total = 0.0
    count = 0

    with ThreadPoolExecutor(max_workers=len(FIXED_CLOB_TOKEN_IDS)) as executor:
        futures = {
            executor.submit(fetch_5m_values_for_token, token_id, start_s, end_s): token_id
            for token_id in FIXED_CLOB_TOKEN_IDS
        }
        for future in as_completed(futures):
            p_vals = future.result()
            if not p_vals or len(p_vals) < expected_count:
                continue
            for p in p_vals:
                total += p
                count += 1

    if count == 0:
        return None

    return float(total / count)


"""
Testanrop.
"""
if __name__ == "__main__":
    """
    Få statistik för medelvärdet som används för backtestet
    """
    max_days = 90

    dates = deque()
    sent_scores = deque()

    for shift in range(max_days):
        score = compute_combined_sentiment(shift, max_days, lookback_minutes=1440)
        print(score)
        if score is not None:
            days_ago = max_days - shift - 1
            date = datetime.now() - timedelta(days=days_ago)
            dates.append(date)
            sent_scores.append(score)

    plt.figure()
    plt.plot(dates, sent_scores, marker="o")
    plt.xlabel("Datum")
    plt.ylabel("Sentiment score Medel")
    plt.title(f"Bitcoin Sentiment över de senaste {len(dates)} dagarna")
    plt.gcf().autofmt_xdate()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
