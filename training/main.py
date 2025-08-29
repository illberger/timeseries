# /training/main.py
"""
Notera att denna fil har alla möjliga anrop och funktioner.
Mycket av gamla variabler etc har även inte städats bort.

De facto parametrarna (batch_size, shift, downsample etc..) finns i den *senaste* oracle.json.
För att initialisera det *avsedda* nätverket, anropa "build_model_hyper_log" (6 i menyn),
de flesta hyperparametrar berörande nätverket är överskridna i denna funktion.

Ps. Labeln är, trots namnet, inte en "log return".
"""
import os
import random
from collections import defaultdict

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

feature_names = ['OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'TimeSin', 'TimeCos', 'Sentiment']  # purely for illustration
feature_n = 8
sequence_length = 48  # is HP

label_width = 1
metrics = ['mae']
EPOCHS = 100 # mainly unused
PATIENCE = 8
BATCH_SIZE = 32  # is HP
SHIFT = 12  # is HP
TUNE_EPOCHS = 200 # unused
TRAIN_SPLIT = 0.6153846154
VAL_SPLIT = 0.2307692308

# model version
modelv_number = 3

# basetuner dir to parse if any
ktv_number = 10


initial_hps = kt.HyperParameters()
#initial_hps.Choice("batch_size", [16, 32, 64, 128])
initial_hps.Int("shift", 3, 16)
initial_hps.Choice("seq_len", [48, 96, 144, 192, 240, 288, 576, 1152])
initial_hps.Choice("downsample", [2, 4, 6, 8, 10])
fetcher = DBFetcher(feature_n)

probe_layers = None
probe_n = 8


def default_baseline_fn(y_true):
    return tf.zeros_like(y_true)


class SignalExtractionLogger(tf.keras.callbacks.Callback):
    """
    Logs: R2 vs baseline, per-layer weight norms, and activation stats.
    Works with model.fit() as-is. Gradient norms available via Option B.
    """

    def __init__(self, val_ds,
                 baseline_fn=default_baseline_fn,
                 layers_to_probe_arg=None,
                 probe_n_arg=None,
                 probe_types=(tf.keras.layers.LSTM, tf.keras.layers.Dense),
                 activation_batches=1):
        super().__init__()
        self.val_ds = val_ds
        self.baseline_fn = baseline_fn
        self.probe_n = probe_n_arg
        self.probe_types = probe_types  # keras layer types
        self.layers_to_probe = layers_to_probe_arg  # names of layer objects (string[])
        self.activation_batches = activation_batches
        self.history = defaultdict(list)
        self._probe_model = None

    def __deepcopy__(self, memo):
        new = self.__class__(
            val_ds=self.val_ds,
            baseline_fn=self.baseline_fn,
            layers_to_probe_arg=self.layers_to_probe,
            probe_n_arg=self.probe_n,
            probe_types=self.probe_types,
            activation_batches=self.activation_batches,
        )
        new.history = defaultdict(list)  # fresh store
        new._probe_model = None
        new._probe_names = None
        return new

    def _build_probe_model(self):
        if self.layers_to_probe is None:
            layers = [l for l in self.model.layers if isinstance(l, self.probe_types)]
            layers = layers[-self.probe_n:] if self.probe_n else layers
        else:
            name2layer = {l.name: l for l in self.model.layers}
            layers = [(item if isinstance(item, tf.keras.layers.Layer) else name2layer[str(item)]) for item in
                      self.layers_to_probe]
        if layers:
            self._probe_model = tf.keras.Model(self.model.inputs, [l.output for l in layers])
            self._probe_names = [l.name for l in layers]

    @staticmethod
    def _layer_weight_norms(layer):
        norms = {}
        for w in layer.weights:
            try:
                norms[w.name] = tf.linalg.global_norm([w]).numpy()
            except Exception:
                pass
        return norms

    def on_train_begin(self, logs=None):
        self._build_probe_model()

    def on_epoch_end(self, epoch, logs=None):
        mse_model = tf.keras.metrics.Mean()
        mse_base = tf.keras.metrics.Mean()

        for x_val, y_val in self.val_ds:
            y_pred = self.model(x_val, training=False)
            base_pred = self.baseline_fn(y_val)
            mse_model.update_state(tf.reduce_mean(tf.square(y_val - y_pred)))
            mse_base.update_state(tf.reduce_mean(tf.square(y_val - base_pred)))

        mse_model_v = float(mse_model.result().numpy())
        mse_base_v = float(mse_base.result().numpy())
        r2_vs_base = 1.0 - (mse_model_v / (mse_base_v + 1e-12))

        self.history['val_mse_model'].append(mse_model_v)
        self.history['val_mse_baseline'].append(mse_base_v)
        self.history['r2_vs_baseline'].append(r2_vs_base)

        layer_norms = {}
        for layer in self.model.layers:
            norms = self._layer_weight_norms(layer)
            if norms:
                layer_norms[layer.name] = norms
        self.history['weight_norms'].append(layer_norms)

        act_stats = {}
        if self._probe_model is not None:
            collected = 0
            for x_val, _ in self.val_ds:
                acts = self._probe_model(x_val, training=False)
                if not isinstance(acts, (list, tuple)):
                    acts = [acts]
                for name, a in zip(self._probe_names, acts):
                    a = tf.convert_to_tensor(a)
                    flat = tf.reshape(a, [tf.shape(a)[0] * tf.shape(a)[1], -1]) if a.shape.rank >= 3 else tf.reshape(a,
                                                                                                                     [
                                                                                                                         tf.shape(
                                                                                                                             a)[
                                                                                                                             0],
                                                                                                                         -1])
                    mean = tf.reduce_mean(flat, axis=0)
                    var = tf.math.reduce_variance(flat, axis=0)
                    # Summaries across units
                    act_stats.setdefault(name, {})
                    act_stats[name].setdefault('unit_mean_mean', []).append(float(tf.reduce_mean(mean).numpy()))
                    act_stats[name].setdefault('unit_var_mean', []).append(float(tf.reduce_mean(var).numpy()))
                collected += 1
                if collected >= self.activation_batches:
                    break
            for lname in act_stats:
                for k in act_stats[lname]:
                    act_stats[lname][k] = float(np.mean(act_stats[lname][k]))
        self.history['activation_stats'].append(act_stats)
        logs = logs or {}
        logs['r2_vs_baseline'] = r2_vs_base
        print(f"\n[SignalExtraction] epoch={epoch} R2_vs_baseline={r2_vs_base:.4f} "
              f"(MSE_model={mse_model_v:.6f}, MSE_base={mse_base_v:.6f})")


class DataTuningHyperband(kt.BayesianOptimization):
    """
    Superbuggig klass i denna version av tensorflow. Gör man något litet "fel" i superklassens keywordarguments så
    skrivs dessa över i sin egna implementering.
    """

    def run_trial(self, trial, *args, **kwargs):
        #bs = trial.hyperparameters.get("batch_size")
        bs = 16
        sh = trial.hyperparameters.get("shift")
        #sh = 12
        seq_len = trial.hyperparameters.get("seq_len")
        downsample = trial.hyperparameters.get("downsample")
        train_ds, val_ds, _, total_seq = make_datasets(shift=sh, batch_size=bs, seq_len=seq_len, downsample=downsample)

        signal_cb = SignalExtractionLogger(
            val_ds=val_ds,
            baseline_fn=default_baseline_fn,
            layers_to_probe_arg=probe_layers,
            probe_n_arg=probe_n,
            activation_batches=1
        )

        user_cbs = list(kwargs.pop("callbacks", []))
        cbs = user_cbs + [signal_cb]
        return super().run_trial(
            trial,
            train_ds,
            validation_data=val_ds,
            callbacks=cbs,
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
                n_feat: int = feature_n,
                _label_width: int = label_width,
                reg: float = 1e-4,
                dropout: float = 0.0):
    """
    :param seq_len:
    :param n_feat:
    :param _label_width:
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
    outputs = keras.layers.Dense(_label_width, activation="linear", name="yhat")(x)
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


def make_datasets(shift: int, batch_size: int, seq_len: int, downsample: int):
    fetcher.set_downsample(downsample)
    total_rows = fetcher.row_count()
    total_sequences = (total_rows - (seq_len + label_width)) // shift + 1
    train_size = int(total_sequences * TRAIN_SPLIT)
    val_size = int(total_sequences * VAL_SPLIT)
    test_size = total_sequences - train_size - val_size
    print(f"Total sequences: {total_sequences}\n"
          f"train: {train_size}\n"
          f"val: {val_size}\n"
          f"test: {test_size}")
    raw_ds = (fetcher.get_dataset()
              .prefetch(tf.data.AUTOTUNE)
              .window(seq_len + label_width,
                      shift=shift,
                      drop_remainder=True))
    full_ds = (raw_ds
               .flat_map(lambda w: w.batch(seq_len + label_width))
               .map(preprocess_window, num_parallel_calls=tf.data.AUTOTUNE)
               .shuffle(buffer_size=10000, seed=42)
               .prefetch(tf.data.AUTOTUNE))
    gap = seq_len // shift

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


# Note: Tuning seq_len; overriden all other HP:s!
def build_model_hyper_log(hp):
    seq_len = hp.get("seq_len")
    inp = keras.Input(shape=(seq_len, feature_n))
    x = inp
    num_lstm = 5  # hp.Int("num_lstm")
    num_dense = 3  # hp.Int("num_dense")
    #use_attn = hp.Boolean("use_attention")
    use_attn = False
    for i in range(num_lstm):
        #units = hp.Int(f'{i}_lstm_units', 32, 256, step=16)
        #drop = hp.Float(f'{i}_lstm_dropout', 0.0, 0.5, step=0.1)
        #reg = hp.Float(f'{i}_lstm_kernel_l2', 1e-6, 1e-2)
        match i:
            case 0:
                units = 176
                drop = 0.0
                reg = 0.002107499708644971
            case 1:
                units = 128
                drop = 0.1
                reg = 0.007421430657090328
            case 2:
                units = 176
                drop = 0.4
                reg = 0.009660633251495054
            case 3:
                units = 64
                drop = 0.3
                reg = 0.00039975200219051396
            case 4:
                units = 224
                drop = 0.2
                reg = 0.009788558058276602

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

    # Dense head
    for i in range(num_dense):
        #units = hp.Int(f'{i}_neurons', 16, 256, step=16)
        use_do = False
        match i:
            case 0:
                units = 64
            case 1:
                units = 208
            case 2:
                units = 240
                use_do = True
        x = keras.layers.Dense(
            units,
            activation='relu',
            name=f"{i}_non_lin_dense"
        )(x)
        #use_do = hp.Boolean(f"{i}_use_dropout")
        if use_do:
            #rate = hp.Float(f"{i}_dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
            rate = 0.4
            x = keras.layers.Dropout(rate, name=f"{i}_dropout")(x)

    out = keras.layers.Dense(label_width, activation='linear')(x)
    model = keras.Model(inputs=inp, outputs=out)

    #lr = hp.Float('learning_rate', 1e-4, 8e-4, sampling='log')
    lr = 0.0002347830559127557
    #delta = hp.Float('huber_delta', 0.02, 0.2)
    delta = 0.07572536051676752
    #alpha = hp.Float('alpha', 0.01, 0.1)
    alpha = 0.011786548551798726
    epochs_decay_frac = 6
    #bs = hp.get("batch_size")
    bs = 16
    shift = hp.get("shift")
    #shift = 12

    ds = hp.get("downsample")
    fetcher.set_downsample(ds)

    total_rows = fetcher.row_count()
    total_seq = (total_rows - (seq_len + label_width)) // shift + 1

    steps_per_epoch = int((total_seq * TRAIN_SPLIT) / bs)
    steps = int(steps_per_epoch * (TUNE_EPOCHS / epochs_decay_frac))

    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr,
                                                   decay_steps=steps,
                                                   alpha=alpha)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_function = tf.keras.losses.Huber(delta=delta)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['mae', 'mse']
    )
    return model


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
                train_ds, val_ds, test_ds, _ = make_datasets(shift=SHIFT, batch_size=BATCH_SIZE,
                                                             seq_len=sequence_length, downsample=6)
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
                    objective=kt.Objective("r2_vs_baseline", direction="max"),
                    max_trials=30,
                    directory=f"kt_{ktv_number}",
                    project_name="log_return",  # "log_return", "kt_7". The model used for report.pdf
                    hyperparameters=initial_hps,
                    seed=42,
                )

                tuner.search(
                    epochs=20,
                    callbacks=callbacks,  # r^2 comparison is inserted in child class
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
                #best_bs = best_hp.get("batch_size")
                best_sh = best_hp.get("shift")
                #est_sh = 12
                best_bs = 16
                #best_lr = best_hp.get("learning_rate")
                best_lr = 0.0002347830559127557
                #best_frac = best_hp.get("epoch_decay_frac")
                best_frac = 6
                #best_hub_delta = best_hp.get("huber_delta")
                best_hub_delta = 0.07572536051676752
                #target_frac = best_hp.get("alpha")
                target_frac = 0.011786548551798726
                seq_len = best_hp.get("seq_len")
                downsample = best_hp.get("downsample")

                train_ds, val_ds, test_ds, tot_seq = make_datasets(shift=best_sh, batch_size=best_bs, seq_len=seq_len,
                                                                   downsample=downsample)
                steps_per_epoch = int((tot_seq * TRAIN_SPLIT) / best_bs)
                steps = int(steps_per_epoch * (EPOCHS / best_frac))
                best_lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=best_lr,
                                                                    decay_steps=steps,
                                                                    alpha=target_frac)

                model.summary()

                signal_cb = SignalExtractionLogger(
                    val_ds=val_ds,
                    baseline_fn=default_baseline_fn,
                    layers_to_probe_arg=probe_layers,
                    probe_n_arg=probe_n,
                    activation_batches=1
                )

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
                    callbacks=callbacks + [signal_cb],
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
                    ckpt_dir = f"../Backtest/files/ckpts_{modelv_number}/offline_final"
                    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
                    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)
                    manager.save()
                    model.save(f"../Backtest/files/final_model_{modelv_number}")
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
