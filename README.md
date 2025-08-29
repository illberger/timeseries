# Projekt för Deep Learning UMU
### Kort om projektet
<code>Tensorflow v2.10.1</code> & <code>NumPy v1.23.5</code>.

#### Huvudvärk gällande <code>/Backtest</code>
Notera i <code>model_manager.predict_close()</code> att sekvenserna, i.e. <code>last_unscaled, X_unscaled</code> "laggar" med <code>offline_shift</code>.
I <code>monitor_predictions</code> beräknas däremot residualen för varje <b>tidsteg</b>, vilket råkar vara 30 minuter. 

### Kom igång med testkörning

Om man endast vill prova göra inferensdelen,
så behöver man endast <code>pip install binance-connector</code>
för själva websocket-grejerna. Notera att variabelnamn är tvetydiga då logiken förändrats sedan start


För att träna en identisk modell kan man se över <code>db_fetcher.query</code> för datat som används. I <code>/fetching/..</code>
finns all nödvänding kod för att hämta träningsdatat (som kräver eventuell behandling, samt lagring). Finns även tillstånd från min tuningsession
i <code>/training/kt_5/..</code> där den "bästa" trial:en kan laddas in, vilket bör resultera i ett likadant nätverk som den bifogade <code>.keras</code>-filen.

