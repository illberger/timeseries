# polymarket_getter.py.py
import json

import httpx


def get_clob_tokens_for_condition(condition_id: str) -> list[str] | None:
    """
    Hämtar market‐objektet för given condition_id och returnerar
    en lista av clobTokenIds (som strängar). Returnerar None om inget hittas.
    """
    url    = "https://gamma-api.polymarket.com/markets"
    params = {"condition_ids": condition_id}

    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        markets = resp.json()
    except Exception as e:
        print(f"[get_clob_tokens_for_condition] Fel vid GET /markets: {e}")
        return None

    if not markets:
        print(f"Inget market hittades för condition_id={condition_id}")
        return None

    m = markets[0]
    raw = m.get("clobTokenIds")
    if not raw:
        print(f"Market‐objekt saknar 'clobTokenIds' för condition_id={condition_id}")
        return None

    try:
        token_list = json.loads(raw)
    except Exception as e:
        print(f"Fel när vi försöker json.loads('clobTokenIds'): {e}")
        return None

    return token_list  # lista av token‐ID som strängar


def main():
    """
    "cond_id" som behövs kodas in kan hämtas genom att gå in på polymarket.com/event/nagon-sorts-prediktions-marknad
    Hämta dessa via devtools i valfri webbläsare. Klienten gör en request med en parameter: market=<cond_id>&limit=30

    Backtestet blir otroligt mycket segare för varje ytterligare sentimentserie
    :return:
    """
    cond_id_BTC1 = "0xd8b9ff369452daebce1ac8cb6a29d6817903e85168356c72812317f38e317613"     # Will Bitcoin reach $1,000,000 by Dec 31, 2025? Avbl. Dec 31 2024....
    cond_id_BTC2 = "0x80026f98f9de40aea8dba02798c4f0067942bba401fa3715209ee7c27482640b"     # US National BTC reserve in 2025? Avbl. Jan 1 2025...
    cond_id_BTC3 = "0x19ee98e348c0ccb341d1b9566fa14521566e9b2ea7aed34dc407a0ec56be36a2"     # MicroStrategy sells any BTC in 2025? # Obs! "Ja"-priset är BTC-bearish. Avbl. Jan 1 2025 ....
    cond_id_BTC4 = "0x324283263c83e789fc092ed8d3333aa93ecc6ef0ba5479db05bbcb2471c92d01"     # Solana all time high by June 30? Avbl. Jan 22 2025...
    cond_id_BTC5 = "0x50b8b0f741566a420756d975faaf4cc4716229cd5183c3a35b3b20ac2b5050ef"     # Will a new country buy Bitcoin in 2025? Avbl. Jan 1 2025 ....
    cond_id_BTC6 = "0x322ba30e34f7ca9c6d43d00660aece6db8f3188a22795df2e3bf28c863a0c4b4"     # BTC Up or Down in Q2? Avbl. Apr 1 2025 ---...


    # 2024. YES på alla.
    cond_id_BTC7 = "0x9c66114d2dfe2139325cc7a408a5cd5d2e73b55d919e2141b3a0ed83fc15895d"  # Will BTC hit $100k in 2024? Avbl. from 2024-03-05 till 2024-12-05
    cond_id_BTC8 = "0xc74c7e76a0d354a27c1cf1b562686e7dc985e2df6762c2c9f5e81ee00448b755"  # Bitcoin New ATH in 2024? Avbl. from 2024-09-21 till 2024-11-05
    cond_id_BTC9 = "0x0ddefa0441efdeca7fe2787c96d710524e648ae8857352f4a52b2ce2629961bf"  # Will MicroStrategy purchase more BTC in 2024? Avbl. frm 2024-09-21 -> 2024-11-05
    cond_id_BTC10 = "0x3272855930be35f026b5de8024d0917b344fb5c8e69a8a8ac09c23167cc9e91b"  # US government BTC reserves in 2024? Avbl. from 2024-07-13 -> 2025-01-01




    condition_ids = [cond_id_BTC10]

    for cond_id in condition_ids:
        print(f"\n=== Processing condition_id={cond_id} ===")
        clob_tokens = get_clob_tokens_for_condition(cond_id)
        if not clob_tokens:
            print(f"Kunde inte hämta några clobTokenIds för condition {cond_id}. Hoppar över.")
            continue

        # Första clob_token_id antas vara för "YES". Verifiera om det finns flera svar.
        token_id = clob_tokens[0]  # Om man tittar på en bearish marknad så ska man ta det 1:a indexet om det är ja/nej fråga. Det får man bedöma själv
        print(f"Clob token id = {token_id}")



if __name__ == "__main__":
    main()
