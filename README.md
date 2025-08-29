# Projekt för Deep Learning UMU
### Kort om projektet
<code>Tensorflow v2.10.1</code> & <code>NumPy v1.23.5</code>.

#### Huvudvärk gällande <code>/Backtest</code>
Notera i <code>model_manager.predict_close()</code> att sekvenserna, i.e. <code>last_unscaled, X_unscaled</code> "laggar" med <code>offline_shift</code>.
I <code>monitor_predictions</code> beräknas däremot residualen för varje <b>tidsteg</b>, vilket råkar vara 30 minuter. 

### Kom igång med testkörning

Om man vill prova göra inferensdelen i <code>/Backtest/</code>,
så behöver man endast <code>pip install binance-connector</code>
för själva websocket-grejerna. Själva datat hämtas under exekvering, så inget förarbete är nödvändigt.


För att träna en identisk modell kan man se över <code>/training/db_fetcher.query</code> för datat som används, alternativt de scheman i <code>/SQL/..</code>.
I <code>/fetching/..</code> finns all nödvänding kod för att hämta träningsdatat (som kräver eventuell behandling, samt lagring). Det finns även tillstånd från en tuningsession
(<code>pip install keras-tuner</code>) i <code>/training/kt_i/..</code> där den "bästa" trial:en kan laddas in, vilket bör resultera i ett likadant nätverk som finns i <code>/Backtest/files..</code>.
Det kan vara lite oordning gällande kardinaliteten och samplingfrekvensen mellan <code>/Backtest</code> och <code>/training</code> då dessa är hårdkodade, så bäst att ladda ned filerna från den ursprungliga commit:en om man vill göra <code>model.fit()</code>. 

