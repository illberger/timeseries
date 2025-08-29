SEQ_LEN = 48
WINDOW_LENGTH_DAY = 288
CLOSE_IDX = 3


OFFLINE_PATH_SENT = "files/final_model"
ONLINE_PATH_SAVE = "files/online_model.keras"
CHECKPOINT = "files/ckpts/offline_final"


FIXED_CLOB_TOKEN_IDS = [
    # BTC Price EoY? (YES), resolved if $1M USDT/USDC On Binance
    "112540911653160777059655478391259433595972605218365763034134019729862917878641",

    # US National BTC reserve in 2025? (Yes)
    "83894672511259544049673946661753374355328822374216474995072428966535091173758",

    # Will a new country buy Bitcoin in 2025? (Yes)
    "52696967762983156376661808083218380818225074723063198070857375460800745709299",

    # BTC up or down in Q2 2025? (Yes) Resolved at end of Q2.
    "45956246277175804727891136697450869076742282101236359440190169889077896442731",
]
"""
Source of "sentiment"-data\n
Each of these prediction-markets give at a point in time (with a fidelity of 5 min) a value between [0,1], representing
the prediction-markets' YES/NO prediction for an 'outcome'. By default, the web-API polled will yield the market's
estimation for the 'outcome' which will automatically resolve this prediction-market (i hope)
\n
"1. Market Creation
Polymarket users can create prediction markets around any real-world event that they choose. 
To do so, they propose an event, such as “Will [Event] occur by [Date]?” The market creator defines the possible
outcomes (e.g., Yes or No) and adds details to clarify conditions for resolution.
Other users can then participate by placing bets on the outcome they believe is most likely.
\n
2. Polymarket Odds
As bets are placed, Polymarket automatically adjusts odds based on the market’s activity. If more users bet on one 
outcome, the odds will reflect a higher likelihood of that event happening. As a result, this dynamic odds system 
ensures fairness and reflects the collective sentiment of all market participants."\n
src=https://www.webopedia.com/crypto/learn/how-does-polymarket-work/
"""