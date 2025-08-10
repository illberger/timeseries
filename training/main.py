# /training/main.py
import os
import random

import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.layers import TimeDistributed, Dense
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from db_fetcher import DBFetcher
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
#import tensorflow_addons as tfa
from data_processor import preprocess_window
import pandas as pd
import numpy as np
from line_profiler import LineProfiler, profile

from training.DBPlotFetcher import DBPlotFetcher
from training.DBPlotFetcherSentiment import DBPlotFetcherSentiment
from training.DbPlotFetcherSinCos import DBPlotFetcherSinCos
from tensorflow.keras.losses import Huber

os.environ['PYTHON'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

feature_names = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'TimeSin', 'TimeCos', 'Sentiment']
feature_n = 8
sequence_length = 48
label_width = 1
metrics = ['mae']
EPOCHS = 100
PATIENCE = 12
BATCH_SIZE = 32
SHIFT = 12
TUNE_EPOCHS = 200
TRAIN_SPLIT = 0.6153846154
VAL_SPLIT = 0.2307692308

initial_hps = kt.HyperParameters()
initial_hps.Choice("batch_size", [16, 32, 64, 128])
initial_hps.Int("shift", 8, 16)
fetcher = DBFetcher()


class DataTuningHyperband(kt.BayesianOptimization):
    """
    Superbuggig klass i denna version av tensorflow. Gör man något litet "fel" i superklassens keywordarguments så
    skrivs dessa över i sin egna implementering.
    """
    def run_trial(self, trial, *args, **kwargs):
        bs = trial.hyperparameters.get("batch_size")
        sh = trial.hyperparameters.get("shift")
        train_ds, val_ds, _, total_seq = make_datasets(shift=sh, batch_size=bs)
        return super().run_trial(
            trial,
            train_ds,
            validation_data=val_ds,
            **kwargs
        )


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, verbose=1, restore_best_weights=True)
]


@profile
def build_model(seq_len: int = sequence_length,
                n_feat: int = 8,
                label_width: int = 1,
                reg: float = 1e-4,
                dropout: float = 0.0):
    """
    :param seq_len:
    :param n_feat:
    :param label_width:
    :param reg:
    :param dropout:
    :return:
    """

    inputs = keras.Input(shape=(seq_len, n_feat))
    x = keras.layers.LSTM(
        112,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(reg),
        dropout=dropout,
        name="lstm_0")(inputs)

    """
    x = keras.layers.LSTM(
        128,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(reg),
        name="lstm_1")(x)
    """
    """
    x = keras.layers.LSTM(
        112,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(reg),
        recurrent_dropout=0.0,
        name="lstm_2")(x)
    x = keras.layers.LSTM(
        112,
        return_sequences=False,
        name="lstm_3")(x)
    """
    #x = keras.layers.LayerNormalization(name="ln_0")(x)
    #x = keras.layers.Dropout(dropout)(x)
    #x = keras.layers.LSTM(
    #    64,
    #     return_sequences=True,
    #     kernel_regularizer=keras.regularizers.l2(reg),
    #     recurrent_dropout=0.00,
    #     name="lstm_1")(x)
    #x = keras.layers.LayerNormalization(name="ln_1")(x)
    #x = keras.layers.Dropout(dropout)(x)
    #x = keras.layers.LSTM(
    #     32,
    #     return_sequences=False,
    #     kernel_regularizer=keras.regularizers.l2(reg),
    #     name="lstm_2")(x)
    #x = keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(reg))(x)
    outputs = keras.layers.Dense(label_width, activation="linear", name="yhat")(x)
    model = keras.Model(inputs, outputs, name="stacked_lstm_welfords")
    init_lr = 1e-4
    opt = tf.keras.optimizers.Adam(learning_rate=init_lr)
    huber_delta = 0.0065
    #loss_function = Huber(delta=huber_delta)
    loss_function = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=opt,
                  loss=loss_function,
                  metrics=["mae", "mse"])
    return model


def make_datasets(shift, batch_size):
    total_rows = fetcher.row_count()
    total_sequences = (total_rows - (sequence_length + label_width)) // shift + 1
    train_size = int(total_sequences * TRAIN_SPLIT)
    val_size = int(total_sequences * VAL_SPLIT)
    test_size = total_sequences - train_size - val_size
    print(f"Total sequences: {total_sequences}\n"
          f"train: {train_size}\n"
          f"val: {val_size}\n"
          f"test: {test_size}")
    raw_ds = (fetcher.get_dataset()
              .prefetch(tf.data.AUTOTUNE)
              .window(sequence_length + label_width,
                      shift=shift,
                      drop_remainder=True))
    full_ds = (raw_ds
               .flat_map(lambda w: w.batch(sequence_length + label_width))
               .map(preprocess_window, num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(buffer_size=10000, seed=42)
               .prefetch(tf.data.AUTOTUNE))
    gap = sequence_length // shift

    train_ds = (full_ds
                .take(train_size)
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (full_ds
              .skip(train_size + gap)
              .take(val_size)
              .batch(batch_size, drop_remainder=True)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (
        full_ds
        .skip(train_size + gap + val_size + gap)
        .take(test_size)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds, test_ds, total_sequences


def build_model_hyper(hp):
    inp = keras.Input(shape=(sequence_length, feature_n))
    x = inp
    num_lstm = hp.Int('num_lstm_layers', 1, 5)
    num_dense = hp.Int('num_dense_layers', 0, 5)
    use_attn = hp.Boolean("use_attention")
    for i in range(num_lstm):
        units = hp.Int(f'{i}_lstm_units', 32, 128, step=16)
        drop = hp.Float(f'{i}_lstm_dropout', 0.0, 0.5, step=0.1)
        reg = hp.Float(f'{i}_lstm_kernel_l2', 1e-5, 1e-3)
        #rec_drop = hp.Float(f'{i}_lstm_recurrent_dropout', 0.0, 0.5, step=0.1)
        return_seq = use_attn or (i < num_lstm - 1)
        x = keras.layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=drop,
            kernel_regularizer=keras.regularizers.l2(reg),
        )(x)

    if use_attn:

        num_heads = hp.Int("attn_heads", 1, 8)
        key_dim = hp.Int("attn_key_dim", 16, 64, step=16)
        attn_drop = hp.Float("attn_dropout", 0.0, 0.5, step=0.1)


        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attn_drop,
            name="self_attn"
        )(x, x)


        x = keras.layers.Add(name="attn_residual")([x, attn_out])
        x = keras.layers.LayerNormalization(name="attn_norm")(x)

        ffn_units = hp.Int("attn_ffn_units", 32, 128, step=32)
        ffn_drop = hp.Float("attn_ffn_dropout", 0.0, 0.5, step=0.1)
        ffn = keras.Sequential([
            keras.layers.Dense(ffn_units, activation='relu'),
            keras.layers.Dropout(ffn_drop),
            keras.layers.Dense(x.shape[-1]),
        ], name="attn_ffn")(x)
        x = keras.layers.Add(name="ffn_residual")([x, ffn])
        x = keras.layers.LayerNormalization(name="ffn_norm")(x)
        x = keras.layers.GlobalAveragePooling1D(name="attn_pool")(x)
    for i in range(num_dense):
        units = hp.Int(f'{i}_neurons', 16, 128, step=16)
        x = keras.layers.Dense(
            units,
            activation='relu',
            name=f"{i}_non_lin_dense"
        )(x)
        use_do = hp.Boolean(f"{i}_use_dropout")
        if use_do:
            rate = hp.Float(f"{i}_dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
            x = keras.layers.Dropout(rate, name=f"{i}_dropout")(x)

    out = keras.layers.Dense(label_width, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)

    lr = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
    alpha = hp.Float('alpha', 0, 1)
    epochs_decay_frac = hp.Int('epoch_decay_frac', 1, 6)
    bs = hp.get("batch_size")
    total_rows = fetcher.row_count()
    total_seq = (total_rows - (sequence_length + label_width)) // (hp.get("shift")) + 1
    steps_per_epoch = int((total_seq * TRAIN_SPLIT) / bs)
    steps = int(steps_per_epoch * (TUNE_EPOCHS / epochs_decay_frac))
    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr,
                                                   decay_steps=steps,
                                                   alpha=alpha)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #loss_function = tf.keras.losses.MeanSquaredError()
    loss_function = tf.keras.losses.Huber(delta=0.05)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae', 'mse']
    )
    return model


def build_model_hyper_log(hp):
    inp = keras.Input(shape=(sequence_length, feature_n))
    x = inp
    num_lstm = 5
    num_dense = 3
    use_attn = hp.Boolean("use_attention")
    for i in range(num_lstm):
        units = hp.Int(f'{i}_lstm_units', 32, 256, step=16)
        drop = hp.Float(f'{i}_lstm_dropout', 0.0, 0.5, step=0.1)
        reg = hp.Float(f'{i}_lstm_kernel_l2', 1e-6, 1e-2)
        #rec_drop = hp.Float(f'{i}_lstm_recurrent_dropout', 0.0, 0.5, step=0.1)
        return_seq = use_attn or (i < num_lstm - 1)
        x = keras.layers.LSTM(
            units,
            return_sequences=return_seq,
            dropout=drop,
            kernel_regularizer=keras.regularizers.l2(reg),
        )(x)

    if use_attn:

        num_heads = hp.Int("attn_heads", 1, 8)
        key_dim = hp.Int("attn_key_dim", 16, 64, step=16)
        attn_drop = hp.Float("attn_dropout", 0.0, 0.5, step=0.1)


        attn_out = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attn_drop,
            name="self_attn"
        )(x, x)


        x = keras.layers.Add(name="attn_residual")([x, attn_out])
        x = keras.layers.LayerNormalization(name="attn_norm")(x)

        ffn_units = hp.Int("attn_ffn_units", 32, 128, step=32)
        ffn_drop = hp.Float("attn_ffn_dropout", 0.0, 0.5, step=0.1)
        ffn = keras.Sequential([
            keras.layers.Dense(ffn_units, activation='relu'),
            keras.layers.Dropout(ffn_drop),
            keras.layers.Dense(x.shape[-1]),
        ], name="attn_ffn")(x)
        x = keras.layers.Add(name="ffn_residual")([x, ffn])
        x = keras.layers.LayerNormalization(name="ffn_norm")(x)
        x = keras.layers.GlobalAveragePooling1D(name="attn_pool")(x)
    for i in range(num_dense):
        units = hp.Int(f'{i}_neurons', 16, 256, step=16)
        x = keras.layers.Dense(
            units,
            activation='relu',
            name=f"{i}_non_lin_dense"
        )(x)
        use_do = hp.Boolean(f"{i}_use_dropout")
        if use_do:
            rate = hp.Float(f"{i}_dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
            x = keras.layers.Dropout(rate, name=f"{i}_dropout")(x)

    out = keras.layers.Dense(label_width, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)

    lr = hp.Float('learning_rate', 1e-4, 8e-4, sampling='log')
    delta = hp.Float('huber_delta', 0.02, 0.2)
    alpha = hp.Float('alpha', 0.01, 0.1)
    epochs_decay_frac = 6
    bs = hp.get("batch_size")
    total_rows = fetcher.row_count()
    total_seq = (total_rows - (sequence_length + label_width)) // (hp.get("shift")) + 1
    steps_per_epoch = int((total_seq * TRAIN_SPLIT) / bs)
    steps = int(steps_per_epoch * (TUNE_EPOCHS / epochs_decay_frac))
    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr,
                                                   decay_steps=steps,
                                                   alpha=alpha)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #loss_function = tf.keras.losses.MeanSquaredError()
    loss_function = tf.keras.losses.Huber(delta=delta)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae', 'mse']
    )
    return model

@tf.function
@profile
def pack_row(ot, op, hi, lo, cl, vol, sent):
    """
    Denna använts för model_optimized....sentz
    :param ot:
    :param op:
    :param hi:
    :param lo:
    :param cl:
    :param vol:
    :param sent:
    :return:
    """
    vec = tf.stack([tf.cast(ot, tf.float32), op, hi, lo, cl, vol, sent], axis=0)
    vec.set_shape([7])
    return vec


@tf.function
@profile
def pack_row_unused(ot, op, hi, lo, cl, vol):
    vec = tf.stack([tf.cast(ot, tf.float32), op, hi, lo, cl, vol], axis=0)
    vec.set_shape([6])
    return vec


def main():
    model = None
    print("TF:", tf.__version__, " NumPy:", np.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
    while True:
        _input = input("Choose\n 1: Z plot-Label Comparison\n"
                       "2. Train NEW model\n"
                       "3. Save model to file\n"
                       "4. Plot Sent Feature Comparison\n"
                       "5. Plot Sin/Cos Feature Comparison\n"
                       "6. Try Search for LSTM stack\n"
                       "7. Autocorr raw ClosePrice\n")
        match _input:
            case "1":
                plot_fetcher = DBPlotFetcher()

                times_global, z_global = plot_fetcher.get_global_close_z()
                times_local, z_local = plot_fetcher.get_local_close_z(sequence_length=sequence_length)

                plt.figure(figsize=(12, 6))
                plt.plot(times_global, z_global, label="Global Z-score (Close)", alpha=0.8)
                plt.plot(times_local, z_local, label="Per-window Z (last close)", alpha=0.8)
                plt.xlabel("OpenTime (ms since epoch)")
                plt.ylabel("Z-score")
                plt.title("Global vs. Per-Window Z-scored Close Price")
                plt.legend()
                plt.tight_layout()
                plt.show()
            case "2":

                model = build_model()
                model.summary()

                """
                Statistics of 2024 offline training-data, where these features are globally Z-scored based on these stats
                Feature	Mean	            Std
                Open	69798,2774252613	12710,5393434195
                High	69866,3593024388	12723,9126849939
                Low	    69728,2148171895	12696,3433144373
                Close	69798,6142232149	12710,7582306191
                Volume	115,566799775478	152,728519726516
                Sent	0,30565474246069	0,211723366775655
                """
                train_ds, val_ds, test_ds, _ = make_datasets(shift=SHIFT, batch_size=BATCH_SIZE)
                all_X = []
                for X_batch, _ in train_ds.unbatch().batch(1024):
                    all_X.append(X_batch.numpy())
                all_X = np.vstack(all_X)
                all_X_flat = all_X.reshape(-1, 8)
                pca = PCA(n_components=4)
                pca.fit(all_X_flat)
                PCA_MEAN = tf.constant(pca.mean_.astype(np.float32))
                PCA_COMPONENTS = tf.constant(pca.components_.T.astype(np.float32))

                def apply_pca(X, y):
                    Xc = X - PCA_MEAN
                    Xp = tf.tensordot(Xc, PCA_COMPONENTS, axes=[2, 0])
                    return Xp, y

                train_pca = train_ds.map(apply_pca, num_parallel_calls=tf.data.AUTOTUNE)
                val_pca = val_ds.map(apply_pca, num_parallel_calls=tf.data.AUTOTUNE)
                Xp_all, y_all = [], []
                for Xp_b, y_b in train_pca.unbatch().batch(1024):
                    Xp_all.append(Xp_b.numpy().reshape(-1, 4 * 288))
                    y_all.append(y_b.numpy().ravel())
                Xp_all = np.vstack(Xp_all)
                y_all = np.concatenate(y_all)
                ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1])
                scores = cross_val_score(ridge, Xp_all, y_all,
                                         cv=TimeSeriesSplit(5), scoring="neg_mean_absolute_error")
                print("Ridge on PCs MAE:", -scores.mean())
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    shuffle=False
                )
                mse_metric = tf.keras.metrics.MeanSquaredError()
                mae_metric = tf.keras.metrics.MeanAbsoluteError()
                for X_batch, y_batch in test_ds:
                    naive_preds = tf.expand_dims(X_batch[:, -1, 3], axis=1)
                    mse_metric.update_state(y_batch, naive_preds)
                    mae_metric.update_state(y_batch, naive_preds)
                baseline_mse_all = mse_metric.result().numpy()
                baseline_mae_all = mae_metric.result().numpy()
                print(f"Naiv baseline (hela test_ds) ‒ MSE: {baseline_mse_all:.5f}, MAE: {baseline_mae_all:.5f}")
                results = model.evaluate(test_ds)
                print("Test loss & metrics:", results)
                hist_df = pd.DataFrame(history.history)
                hist_df[['loss', 'val_loss']].plot(figsize=(8, 5))
                plt.title("Training vs Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(["Train", "Val"])
                plt.show()
            case "6":

                tuner = DataTuningHyperband(
                    hypermodel=lambda h: build_model_hyper_log(h),
                    objective="val_mae",
                    max_trials=30,
                    directory="kt_7",
                    project_name="log_return",  # its actually simple return but whatever
                    hyperparameters=initial_hps,
                    seed=42,
                )
                tuner.search(
                    epochs=100,
                    callbacks=callbacks,
                    verbose=1
                )

                best_hps_list = tuner.get_best_hyperparameters(num_trials=3)
                print("Top n hyperparameter sets:")
                for rank, hp in enumerate(best_hps_list, start=1):
                    print(f"\n--- Rank #{rank} ---")
                    for name, val in hp.values.items():
                        print(f"{name:20s}: {val}")

                model = tuner.get_best_models(1)[0]
                best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
                best_bs = best_hp.get("batch_size")
                best_sh = best_hp.get("shift")
                best_lr = best_hp.get("learning_rate")
                #best_frac = best_hp.get("epoch_decay_frac")
                best_frac = 6
                best_hub_delta = best_hp.get("huber_delta")
                target_frac = best_hp.get("alpha")
                train_ds, val_ds, test_ds, tot_seq = make_datasets(shift=best_sh, batch_size=best_bs)
                steps_per_epoch = int((tot_seq * TRAIN_SPLIT) / best_bs)
                steps = int(steps_per_epoch * (EPOCHS / best_frac))
                best_lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=best_lr,
                                                                    decay_steps=steps,
                                                                    alpha=target_frac)

                model.summary()
                optimizer = tf.keras.optimizers.Adam(learning_rate=best_lr)
                loss_function = tf.keras.losses.Huber(delta=best_hub_delta)
                model.compile(
                    optimizer=optimizer,
                    loss=loss_function,
                    metrics=['mae', 'mse']
                )
                history_tuned = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    shuffle=False
                )
                mse_metric = tf.keras.metrics.MeanSquaredError()
                mae_metric = tf.keras.metrics.MeanAbsoluteError()
                for X_batch, y_batch in test_ds:
                    naive_preds = tf.zeros_like(y_batch)
                    mse_metric.update_state(y_batch, naive_preds)
                    mae_metric.update_state(y_batch, naive_preds)
                baseline_mse_all = mse_metric.result().numpy()
                baseline_mae_all = mae_metric.result().numpy()
                print(f"Naiv baseline (hela test_ds) ‒ MSE: {baseline_mse_all:.5f}, MAE: {baseline_mae_all:.5f}")
                results = model.evaluate(test_ds)
                print("Test loss & metrics:", results)
                hist_df = pd.DataFrame(history_tuned.history)
                hist_df[['loss', 'val_loss']].plot(figsize=(8, 5))
                plt.title("Training vs Validation Loss")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend(["Train", "Val"])
                plt.show()

            case "3":
                if model is None:
                    print("No current model to save. Please train a model first.")
                else:
                    ckpt_dir = "ckpts/offline_final"
                    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
                    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
                    manager.save()
                    model.save("../Backtest/files/final_model")
            case "4":

                sent_plot_fetcher = DBPlotFetcherSentiment()
                times_z, z_sent = sent_plot_fetcher.get_global_sentiment_z()
                times_raw, raw_sent = sent_plot_fetcher.get_raw_sentiment()

                plt.figure(figsize=(12, 6))
                plt.plot(times_raw, raw_sent, label="Raw SentimentMean", alpha=0.8)
                plt.plot(times_z, z_sent, label="Global Z-scored Sentiment", alpha=0.8)
                plt.xlabel("PriceTimeMs (ms since epoch)")
                plt.ylabel("Sentiment")
                plt.title("Raw vs. Globally Z-scored SentimentMean")
                plt.legend()
                plt.tight_layout()
                plt.show()
            case "5":
                time_fetcher = DBPlotFetcherSinCos()
                times_raw, sin_raw, cos_raw = time_fetcher.get_raw_sincos()
                times_z, sin_z, cos_z = time_fetcher.get_global_sincos_z()
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                ax1.plot(times_raw, sin_raw, label="sin(raw)", alpha=0.8)
                ax1.plot(times_raw, cos_raw, label="cos(raw)", alpha=0.8)
                ax1.set_ylabel("raw sin/cos")
                ax1.legend()
                ax1.grid(True)
                ax2.plot(times_z, sin_z, label="SinTZ", alpha=0.8)
                ax2.plot(times_z, cos_z, label="CosTZ", alpha=0.8)
                ax2.set_xlabel("OpenTime (ms since epoch)")
                ax2.set_ylabel("Z-scored sin/cos")
                ax2.legend()
                ax2.grid(True)

                plt.suptitle("Raw vs. Global Z-scored Time-of-Day Features")
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
            case "7":
                df = fetcher.fetch_close_series('BTCUSDT')
                autocorr = fetcher.compute_autocorrelation(df['ClosePrice'], nlags=2000)
                plt.figure()
                plt.plot(np.arange(2000 + 1), autocorr)
                plt.axhline(0.2, linestyle="--")
                plt.xlabel("Lag t")
                plt.ylabel(r"$\rho_t$")
                plt.title("Autocorrelation of ClosePrice")
                plt.show()


if __name__ == '__main__':
    main()

    # kernprof -l -v /training.main.py
