# /Backtest/main.py
"""
NumPy v1.23.5
TensorFlow v2.10
Note that this code is highly unoptimized
"""

import logging
import time
from collections import deque
import optuna
from matplotlib import pyplot as plt
import numpy as np
from binance_client import BinanceWebSocketClient
from model_manager import ModelManager, naive_forecast, OptimizerArgs
import tensorflow as tf
import gc
import model_manager
from itertools import product
from typing import Optional
from CONSTANTS import SEQ_LEN, WINDOW_LENGTH_DAY, ONLINE_PATH_SAVE, OFFLINE_PATH_SENT, CHECKPOINT

#logging.basicConfig(level=logging.INFO)  # Avkommentera för extra logging - sentimentfetchingen kan nekas då det är hög kostnad på GET

#predictions = {}
true_records = []
"""
Main statistics/results
"""
DOWNSAMPLE = int(WINDOW_LENGTH_DAY / 48)
"""
Determines n:th row to take
"""
SLIDE = SEQ_LEN * 2
"""
This is the 'window'-length being kept in memory at all times in the instance of BinanceWebSocketClient
"""
lstm_abs_errors = deque(maxlen=SEQ_LEN)
"""
A rolling window of absolute errors for each sequence, shifted by 1 (unaffected by offline_shift).  
"""
naive_abs_errors = deque(maxlen=SEQ_LEN)
"""
A rolling window of absolute errors for each sequence, shifted by 1 (unaffected by offline_shift).  
"""
lstm_scalar_mae = []
"""
A list of absolute errors for each sequence/datapoint predicted so far.
"""
naive_scalar_mae = []
"""
A list of absolute errors for each sequence/datapoint predicted so far.
"""
last_fetch_shift = -1

model_errs_signed = []
naive_errs_signed = []

start_date = None
time_stat = None

IS_SENTIMENT_EMBEDDED = True
OFFLINE_PATH = ""


def dm_test(errors_model, errors_naive, h=1, power=1, lag=SEQ_LEN):
    import numpy as np, math
    e1 = np.asarray(errors_model);
    e2 = np.asarray(errors_naive)
    assert e1.shape == e2.shape == (e1.size,), "length mismatch"
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    elif power == 2:
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) ** power - np.abs(e2) ** power

    T = d.size
    dbar = d.mean()
    dc = d - dbar

    if lag is None:
        lag = max(1, int(round(T ** (1 / 3))))

    gamma0 = np.dot(dc, dc) / T
    s = gamma0
    for k in range(1, min(lag, T - 1) + 1):
        gk = np.dot(dc[:-k], dc[k:]) / T
        w = 1.0 - k / (lag + 1.0)
        s += 2.0 * w * gk
    var_hat = s / T
    scale = 1.0
    if h > 1:
        scale = math.sqrt(T + 1 - 2 * h + (h * (h - 1)) / T) / math.sqrt(T)

    dm_stat = (dbar / math.sqrt(var_hat)) * scale
    from math import erf, sqrt
    cdf = 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0)))
    p_two = 2.0 * (1.0 - cdf)
    return dm_stat, p_two


def mae(e):
    return np.mean(np.abs(e))


def rmse(e):
    return np.sqrt(np.mean(e ** 2))


def subscribe_and_predict(ws_client, model_mng, symbol, fetch_shift, slice_shift, max_days, start_time_ms: int,
                          mae_lstm: Optional, mae_naive: Optional):
    """
    Does one timestep and returns a prediction. Manages window and data.
    :param ws_client:
    :param model_mng:
    :param symbol:
    :param fetch_shift:
    :param slice_shift:
    :param max_days:
    :param start_time_ms:
    :param mae_lstm:
    :param mae_naive:
    :return:
    """
    ret_lstm_mae: Optional = None
    ret_naive_mae: Optional = None
    global last_fetch_shift
    symbol = symbol.upper()
    if fetch_shift > last_fetch_shift and fetch_shift < max_days:
        if IS_SENTIMENT_EMBEDDED:
            series = ws_client.compute_sentiment_series(fetch_shift, max_days, start_time_ms)
            ws_client.sentiment_series = series
        ws_client.closed_candles[symbol] = []
        if IS_SENTIMENT_EMBEDDED:
            while len(ws_client.sentiment_series) < SLIDE:
                time.sleep(0.01)
        ws_client.fetch_historical_5m_candles(symbol, fetch_shift, max_days, start_time_ms, lookback_minutes=1440)
        last_fetch_shift = fetch_shift
        if len(lstm_abs_errors) > 0:
            rolling_mae = sum(lstm_abs_errors) / len(lstm_abs_errors)
            rolling_mae_naive = sum(naive_abs_errors) / len(naive_abs_errors)
            lstm_scalar_mae.append(rolling_mae)
            naive_scalar_mae.append(rolling_mae_naive)
            ret_naive_mae = rolling_mae_naive
            ret_lstm_mae = rolling_mae
            step_x_axis = (SEQ_LEN * fetch_shift) + slice_shift

            assert len(lstm_scalar_mae) <= max_days
            mae_tot_lstm = np.mean(lstm_scalar_mae)
            mae_tot_naive = np.mean(naive_scalar_mae)
            print(
                f"MAE: [ML: {rolling_mae:.6f}, BASELINE: {rolling_mae_naive:.6f}] @X: {step_x_axis}. W: {fetch_shift}/{max_days}."
                f"Full MAE [ML: {mae_tot_lstm:.6f}, BASELINE: {mae_tot_naive:.6f}]")
        while len(ws_client.closed_candles.get(symbol, [])) < SLIDE:
            time.sleep(0.01)

    seq_scaled, _, seq_raw, last_open_ts, sigma_close_vec, mu_close_vec = ws_client.get_latest_sequence(
        symbol,
        slice_shift,
        seq_len=SEQ_LEN
    )
    while seq_scaled is None:
        time.sleep(0.05)
    _, lstm_pred, lr, k = model_mng.predict_close(symbol,
                                                    seq_scaled,
                                                    seq_raw,
                                                    slice_shift,
                                                    fetch_shift,
                                                    max_days,
                                                    mae_lstm,
                                                    mae_naive,
                                                    sigma_close_vec,
                                                    mu_close_vec)
    _, naive_pred = naive_forecast(symbol, seq_raw)

    return last_open_ts, lstm_pred, naive_pred, ret_lstm_mae, ret_naive_mae, lr, k


def monitor_predictions(_ws_client, _model_mng, _symbol, _max_days, _start_time_ms, disable_plots=False):
    """
    Main loop of backtest.\n
    Steps forward, calculates errors and eventual visual data.
    :param _ws_client: Http and Websocket wrapper object
    :param _model_mng: ModelManager Object
    :param _symbol:
    :param _max_days: Total time to *back*test, i.e., the first window starts at (start_time - max_days) + SEQ_LEN
    :param _start_time_ms: Time in unix ms to step TOWARDS
    :param disable_plots:
    :return:
    """
    global time_stat
    fetch_shift = 0
    slice_offset = 0
    time_stat = time.time()
    print(f"Backtest started at {_start_time_ms} for {_max_days} days.\n")

    mae_lstm: Optional = None
    mae_naive: Optional = None
    #lr: Optional = None
    #k: Optional = None
    last_ts, lstm, naive, mae_lstm, mae_naive, _, _ = subscribe_and_predict(_ws_client,
                                                                            _model_mng,
                                                                            _symbol,
                                                                            fetch_shift,
                                                                            slice_offset,
                                                                            _max_days,
                                                                            _start_time_ms,
                                                                            mae_lstm,
                                                                            mae_naive)

    for _ in range(_max_days * SEQ_LEN):

        seq = _ws_client.get_latest_sequence(_symbol, shift=slice_offset + 1, seq_len=SEQ_LEN)

        if not seq or seq[0] is None:
            continue
        seq_scaled, _, seq_raw, current_ts, _, _ = seq

        true_close = float(seq_raw[-1][3])
        true_close_Z_test = float(seq_scaled[-1][3])  # Testing
        if IS_SENTIMENT_EMBEDDED:
            z_sent = float(seq_scaled[-1][7])
        else:
            z_sent = None
        err = abs(true_close - lstm)
        err_naive = abs(true_close - naive)
        #if not model_mng.do_update(slice_shift=slice_offset) and fetch_shift > 0:
        #    lr = None
        #    k = None
        true_records.append({
            'lstm_pred': lstm,
            'naive_pred': naive,
            'true': true_close,
            'true_z': true_close_Z_test,
            'z_sent': z_sent
            #    'lr': lr,
            #    'k': k
        })
        lstm_abs_errors.append(err)
        #model_errs_signed.append(err)

        naive_abs_errors.append(err_naive)
        #naive_errs_signed.append(err_naive)
        slice_offset += 1
        if slice_offset + SEQ_LEN >= len(_ws_client.closed_candles[_symbol]):
            fetch_shift += 1
            slice_offset = 0
        #if model_mnger.do_replay(slice_offset):
        last_ts, lstm, naive, mae_lstm, mae_naive, lr, k = subscribe_and_predict(_ws_client,
                                                                                 _model_mng,
                                                                                 _symbol,
                                                                                 fetch_shift,
                                                                                 slice_offset,
                                                                                 _max_days,
                                                                                 _start_time_ms,
                                                                                 mae_lstm,
                                                                                 mae_naive)
        #else:
        #    continue

    #dm, p = dm_test(model_errs_signed, naive_errs_signed, h=1, power=1, lag=SEQ_LEN)
    #print(f"DM={dm:.3f}, p={p:.4f}")

    #valid = [i for i, r in enumerate(true_records) if r['lr'] is not None]
    idx = np.arange(len(true_records))  # Evenely spaced 1d array of integers
    actuals = np.array([r['true'] for r in true_records])
    lstm_v = np.array([r['lstm_pred'] for r in true_records])
    naive_v = np.array([r['naive_pred'] for r in true_records])
    lstm_err = np.array([r['true'] - r['lstm_pred'] for r in true_records])
    naive_err = np.array([r['true'] - r['naive_pred'] for r in true_records])
    z_test = np.array([r['true_z'] for r in true_records])  # Testing
    sent_z = np.array([r['z_sent'] for r in true_records])  # Testing

    #lr = [true_records[i]['lr'] for i in valid]
    #k = [true_records[i]['k'] for i in valid]

    def mape(e):
        return np.abs(e) / actuals * 100

    mae_lstm = mae(lstm_err)
    mae_naive = mae(naive_err)
    rmse_lstm = rmse(lstm_err)
    rmse_naive = rmse(naive_err)
    mean_mape_lstm = np.mean(mape(lstm_err))
    mean_mape_naive = np.mean(mape(naive_err))

    lstm_mapes = mape(lstm_err)
    naive_mapes = mape(naive_err)
    low_thr_lstm = np.percentile(lstm_mapes, 33)
    high_thr_lstm = np.percentile(lstm_mapes, 66)
    low_thr_naive = np.percentile(naive_mapes, 33)
    high_thr_naive = np.percentile(naive_mapes, 66)

    def categorize(m, low, high):
        return np.where(m <= low, 'green',
                        np.where(m <= high, 'yellow', 'red'))

    colors_lstm = categorize(lstm_mapes, low_thr_lstm, high_thr_lstm)
    colors_naive = categorize(naive_mapes, low_thr_naive, high_thr_naive)
    colors_lstm = colors_lstm.flatten()
    colors_naive = colors_naive.flatten()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 12), sharex=True)

    ax1.plot(idx, actuals, color='gray', linewidth=0.5, label='True')
    ax1.scatter(idx, lstm_v, c=colors_lstm, s=2)
    text1 = (
        f"MAE  {mae_lstm:.2f}   RMSE  {rmse_lstm:.2f}\n"
        f"MAPE  {mean_mape_lstm:.2f}%"
    )
    ax1.text(0.02, 0.95, text1, transform=ax1.transAxes,
             fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    ax1.set_title(f"LSTM för {_symbol}. {_max_days} dagar.")
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle=':', linewidth=0.5)

    ax2.plot(idx, actuals, color='gray', linewidth=0.5, label='True')
    ax2.scatter(idx, naive_v, c=colors_naive, s=2)
    text2 = (
        f"MAE  {mae_naive:.2f}   RMSE  {rmse_naive:.2f}\n"
        f"MAPE  {mean_mape_naive:.2f}%"
    )
    ax2.text(0.02, 0.95, text2, transform=ax2.transAxes,
             fontsize=10, va='top',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    ax2.set_title("Baseline")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Price")
    ax2.grid(True, linestyle=':', linewidth=0.5)

    ax3.plot(idx, sent_z, color='gray', linewidth=0.5, label='Sent')
    ax3.set_title("Sent")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("sent")
    ax3.grid(True, linestyle=':', linewidth=0.5)

    ax4.plot(idx, z_test, color='gray', linewidth=0.7, label='Z-Close')
    ax4.set_title("Z-Close")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Z-Close")
    ax4.grid(True, linestyle=':', linewidth=0.7)

    """
    ax5.plot(valid, lr, color='gray', linewidth=0.5, label='lr')
    ax5.set_title("lr")
    ax5.set_xlabel("Step")
    ax5.set_ylabel("lr")
    ax5.grid(True, linestyle=':', linewidth=0.5)

    # k, testing
    ax6.plot(valid, k, color='gray', linewidth=0.5, label='k')
    ax6.set_title("k")
    ax6.set_xlabel("Step")
    ax6.set_ylabel("k")
    ax6.grid(True, linestyle=':', linewidth=0.5)
    """

    # Label
    global_text = (
        f"Över hela perioden:\n"
        f"LSTM MAE={mae_lstm:.2f}, MAPE={mean_mape_lstm:.2f} %   |   "
        f"Naive MAE={mae_naive:.2f}, MAPE={mean_mape_naive:.2f} %"
    )
    fig.text(0.5, 0.02, global_text,
             ha='center', va='bottom',
             fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.8))

    #if mae_lstm < mae_naive:
    #    _model_mng.save_model()

    if not disable_plots:
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
    else:
        print(f"Baseline MAE: {mae_naive}")
        return {
            'mae_lstm': mae_lstm,
            'mae_naive': mae_naive,
            'rmse_lstm': rmse_lstm,
        }


def load_model():
    """
    :param model_path: Change this param to whatever .keras file and see exceptions (some are stateful)
    :param online_path:
    :return:
    """
    true_path: str = ""
    try:
        # ignore this
        model = tf.keras.models.load_model(ONLINE_PATH_SAVE)
        true_path = ONLINE_PATH_SAVE
    except (ValueError, OSError):
        if IS_SENTIMENT_EMBEDDED:
            path = OFFLINE_PATH_SENT
        else:
            path = OFFLINE_PATH
        model = tf.keras.models.load_model(path, compile=True)
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        latest = tf.train.latest_checkpoint(CHECKPOINT)
        if latest:
            ckpt.restore(latest).expect_partial()
        true_path = path
    print("Model loaded from", true_path)
    return model


def tune() -> None:
    # start_time_ms = int(time.time() * 1000)
    start_time_ms = 1751320800000
    symbol = "BTCUSDT"
    ws = BinanceWebSocketClient(is_sent_feature=IS_SENTIMENT_EMBEDDED)
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )

    offline_shift = 12

    def objective(trial: optuna.trial.Trial):
        max_days = 165

        per_alpha = trial.suggest_float("per_alpha", 0.5, 1.0)  # now mui importante than ever
        per_gamma = trial.suggest_float("per_gamma", 0.15, 1.6)
        sent_i_pct = 70
        zcap = trial.suggest_float("zcap", 2.5, 4.0)
        interval = 1.0
        update_interval = int(SEQ_LEN * interval)
        optimizer_args = OptimizerArgs(
            optimizer_name="Adam",
            new=True,
            ref_lr=1e-06,
            min_lr_frac=0.1,
            lr_scale=2.5,
            per=True,
            forget=False,
            update_interval=update_interval,
            offline_shift=offline_shift,
            max_days=max_days,
        )

        global last_fetch_shift, true_records
        last_fetch_shift = -1
        # sentiment_for_stat.clear()
        true_records.clear()
        lstm_scalar_mae.clear()
        naive_scalar_mae.clear()
        model_errs_signed.clear()
        naive_errs_signed.clear()
        ws.closed_candles[symbol] = []
        model_manager.TOTAL_UPDATES = 0
        tf.keras.backend.clear_session()
        gc.collect()
        model = load_model()
        model_mnger = ModelManager(model=model,
                                   optimizer_args=optimizer_args,
                                   max_days=max_days,
                                   grad_clip=False,
                                   huber_pct=95,
                                   per_alpha=per_alpha,
                                   per_beta_min=0.1,
                                   per_beta_max=0.7,
                                   trend_alpha=0.0278,
                                   trend_lambda=0.6508,
                                   trend_floor=1.882e-5,
                                   trend_gamma=1.634,
                                   perf_scale_beta=0.0574,
                                   batch_size=16,
                                   update_interval=update_interval,
                                   offline_shift=offline_shift,
                                   per=True,
                                   perf_slow=1,
                                   perf_fast=0.05,
                                   sent_gamma=per_gamma,
                                   sent_i_pct=sent_i_pct,
                                   zcap=zcap)
        metrics = monitor_predictions(
            ws, model_mnger, symbol, _max_days=max_days,
            _start_time_ms=start_time_ms,
            disable_plots=True
        )
        return metrics['mae_lstm']

    study.optimize(objective, n_trials=10)
    print("Bästa parametrar:", study.best_params)
    print("Bästa MAE_LSTM :", study.best_value)


def normal_run() -> None:
    models = load_model()
    max_days = 165
    offline_shift = 12
    update_interval = int(SEQ_LEN * 1)
    # int(time.time() * 1000) for systemtime
    time_start = 1751320800000  # 2025-07-01
    symbol = "BTCUSDC"

    optimizer_args = OptimizerArgs(
        optimizer_name="Adam",
        new=True,
        ref_lr=1e-06,
        min_lr_frac=0.1,
        lr_scale=2.5,
        per=True,
        forget=False,
        update_interval=update_interval,
        offline_shift=offline_shift,
        max_days=max_days,
    )
    # See dosctring
    model_mnger = ModelManager(model=models,
                               optimizer_args=optimizer_args,
                               max_days=max_days,
                               grad_clip=False,
                               huber_pct=95,
                               per_alpha=0.78,
                               per_beta_min=0.1,
                               per_beta_max=0.7,
                               trend_alpha=0.0278,
                               trend_lambda=0.6508,
                               trend_floor=1.882e-5,
                               trend_gamma=1.634,
                               perf_scale_beta=0.0574,
                               batch_size=16,
                               update_interval=update_interval,
                               offline_shift=offline_shift,
                               per=True,
                               perf_slow=1,
                               perf_fast=0.05,
                               sent_gamma=1.5,
                               sent_i_pct=75,
                               zcap=3.0)

    ws_client = BinanceWebSocketClient(is_sent_feature=IS_SENTIMENT_EMBEDDED)
    monitor_predictions(ws_client, model_mnger, symbol, max_days, time_start)
    ws_client.stop()


if __name__ == '__main__':
    normal_run()
    #tune()
