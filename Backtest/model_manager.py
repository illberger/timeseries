# /Backtest/model_manager.py

import tensorflow as tf
import numpy as np
from collections import deque
from typing import Optional, final
from CONSTANTS import SEQ_LEN, WINDOW_LENGTH_DAY, CLOSE_IDX
from tensorflow.keras.layers import LSTM
import random

#import tensorflow_addons as tfa  # 0.16.1 - kanske inte kompatibel.

DOWNSAMPLE = int(WINDOW_LENGTH_DAY / 48)
TOTAL_UPDATES = 0
GLOBAL_EPS = 1e-6
TRUST_MIN_APPLY = 1.0
ZERO = 0

IS_SENTIMENT_EMBEDDED = True
"""
deprecated flag
"""


class OptimizerArgs:
    def __init__(
            self,
            optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
            optimizer_name: Optional[str] = None,
            min_lr_frac: float = 0.1,
            lr_scale: float = 1.0,
            per: bool = False,
            new: bool = False,
            forget: bool = False,
            update_interval: int = 1,
            offline_shift: int = 0,
            max_days: int = 1,
            updates_per_day: Optional[int] = None,
            ref_lr: Optional[float] = None,
            **optimizer_kwargs
    ):
        """
        Holds all optimizer init args in one object.
        If `new=True` and `optimizer_name` is provided, a new optimizer
        instance will be created with ref_lr and any optimizer_kwargs.
        TODO: test optimizer with kwargs
        """
        self.optimizer = optimizer
        self.optimizer_name = optimizer_name
        self.min_lr_frac = min_lr_frac
        self.lr_scale = lr_scale
        self.per = per
        self.new = new
        self.forget = forget
        self.update_interval = update_interval
        self.offline_shift = offline_shift
        self.max_days = max_days
        self.updates_per_day = updates_per_day
        self.ref_lr = ref_lr
        self.optimizer_kwargs = optimizer_kwargs


def init_optimizer_from_args(args: OptimizerArgs):
    """
    Wrapper to init optimizer from an OptimizerArgs instance.
    """

    optimizer = args.optimizer
    if args.new:
        if not args.optimizer_name:
            raise ValueError("Must provide optimizer_name when new=True.")
        opt_cls = getattr(tf.keras.optimizers, args.optimizer_name, None)
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer name: {args.optimizer_name}")
        if args.ref_lr is None:
            args.ref_lr = 0.00000275
        optimizer = opt_cls(learning_rate=args.ref_lr, **args.optimizer_kwargs)

    return init_optimizer(
        optimizer=optimizer,
        min_lr_frac=args.min_lr_frac,
        lr_scale=args.lr_scale,
        per=args.per,
        new=args.new,
        forget=args.forget,
        update_interval=args.update_interval,
        offline_shift=args.offline_shift,
        max_days=args.max_days,
        updates_per_day=args.updates_per_day,
        ref_lr=args.ref_lr,
    )


def naive_forecast(symbol: str, X_unscaled: np.ndarray):
    """
    Enkel naiv prognos: antar att nästa close-pris blir samma som senaste.\n
    """
    seq_unscaled = X_unscaled
    if seq_unscaled is None or seq_unscaled.shape[0] == 0:
        return 0.0, None

    last_close = seq_unscaled[-1, CLOSE_IDX]
    y_naive = 0.0
    pred_close_naive = last_close
    return y_naive, pred_close_naive


def ema_alpha_set(N: int):
    N = max(1, int(N))
    alpha = 2 / (N + 1)
    return alpha

def normalized_trend_slope(y, latest_first=False, eps=1e-8):
    """
    y: 1D array-like of sentiment for a single window
    Returns slope/std per timestep (unitless).
    """
    y = np.asarray(y, dtype=np.float32)
    if latest_first:
        y = y[::-1]
    T = y.shape[0]
    t = np.arange(T, dtype=np.float32)
    t_mean = (T - 1) / 2.0
    y_mean = y.mean()
    num = np.dot(t - t_mean, y - y_mean)
    den = np.dot(t - t_mean, t - t_mean) + eps
    slope = num / den

    std_y = y.std() + eps
    return float(slope / std_y)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ewma(arr, tau):
    j = np.arange(len(arr))[::-1]
    ww = np.exp(-j / tau)
    ww /= ww.sum()
    return float((ww * arr).sum())


def _current_lr(optimizer: tf.keras.optimizers.Optimizer):
    lr = optimizer.learning_rate
    if callable(lr):
        return float(lr(optimizer.iterations).numpy())
    return float(lr.numpy())


def _decay_steps_from_updates(updates_per_day: int, max_days: int) -> int:
    return int(updates_per_day * max_days)


def init_optimizer(
        optimizer: tf.keras.optimizers.Optimizer,
        min_lr_frac: float,
        lr_scale: float,
        per: bool,
        new: bool,
        forget: bool,
        update_interval: int,
        offline_shift: int,
        max_days: int,
        updates_per_day: int = None,
        ref_lr: float = None,
):
    """

    :param optimizer: The optimizer used for model.fit()
    :param min_lr_frac:
    :param lr_scale:
    :param per:
    :param new:
    :param update_interval:
    :param offline_shift:
    :param max_days:
    :param updates_per_day:
    :param ref_lr:
    :return:
    """
    if ref_lr is None:
        if new:
            ref_lr = 0.00000275
        else:
            ref_lr = _current_lr(optimizer)

    max_lr_abs = float(ref_lr * lr_scale)
    min_lr_abs = float(max_lr_abs * min_lr_frac)

    """
    if per:
        lr_var = tf.Variable(min_lr_abs, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = lr_var
        return optimizer, min_lr_abs, max_lr_abs
    """
    max_updates_possible = SEQ_LEN / offline_shift
    if updates_per_day is None:
        updates_per_day = max(max_updates_possible, update_interval / SEQ_LEN)

    decay_steps = _decay_steps_from_updates(updates_per_day, max_days)

    alpha_frac = float(min_lr_abs / max_lr_abs)

    new_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=max_lr_abs,
        decay_steps=decay_steps,
        alpha=alpha_frac,
    )

    class _Shifted(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base, start_step):
            self.base = base
            self.start = tf.constant(int(start_step), dtype=tf.int64)

        def __call__(self, step):
            s = tf.maximum(tf.cast(step, tf.int64) - self.start, 0)
            return self.base(s)

    def decay_optimizer_state(opt, factor=0.5):
        for var in opt.variables():
            name = var.name.lower()
            if "iter" in name or "iteration" in name:
                continue
            var.assign(var * factor)
            print(var * factor)

    if forget:
        decay_optimizer_state(optimizer)
    start_step = int(optimizer.iterations.numpy())
    optimizer.learning_rate = _Shifted(new_schedule, start_step=start_step)

    return optimizer, min_lr_abs, max_lr_abs, decay_steps


class ModelManager:
    """
    TODO: Decouple offline_shift from either the call of online_update, or the appending or length of replay,
    the point is that the frequency that which priorities can be redistributed should not be guarded by the timestep-shift of
    each sequence
    """

    def __init__(self, model, optimizer_args: OptimizerArgs, max_days: int, grad_clip: bool,
                 huber_pct: int, per_alpha: float, per_beta_min, per_beta_max,
                 trend_alpha: float, trend_lambda: float, trend_floor: float, trend_gamma: float,
                 perf_scale_beta: float,
                 batch_size: int,
                 update_interval: int, offline_shift: int, per: bool,
                 perf_fast: float,
                 perf_slow: float,
                 sent_gamma: float,
                 sent_i_pct: int,
                 zcap: float):
        """
        :param sent_gamma: Pass 0 to this to disregard Sentiment in the probability term inside the PER algorithm.
        :param update_interval: [Pass 0 to this to never update model. Otherwise, pass it a multiple of SEQ_LEN (it shouldn't require this,
                                but the math is bugged.)] Interval of timesteps to do an update on, with respect to sequence length. I.e.,
                                every update_interval of ((fetch_shift * SEQ_LEN) + slice_shift), do a parameter update.
        :param offline_shift: This parameter does not affect inference. It determines timesteps between sequences in the replay deque.
                        This parameter does however determine n timesteps online_update is called.
        """

        # Initial priority
        self.p_new = None
        self.priority_percentile: final | int = sent_i_pct
        self.zcap = zcap

        self.grad_clip: final | bool = grad_clip
        self.per: final | bool = per
        self.max_batch: final | int = batch_size
        self.update_interval: final | int = update_interval
        self.offline_shift: int = offline_shift
        self.beta_horizon_updates = int((SEQ_LEN / update_interval) * max_days) if (update_interval != 0) else 0
        print(self.beta_horizon_updates)

        self.max_days: final | int = max_days
        self.fast_perf_ema: float | Optional = None
        self.slow_perf_ema: float | Optional = None

        self.const_label_index = SEQ_LEN - SEQ_LEN

        # exponent for moving average of performance metrics
        # Note that N now needs to be thought of as timesteps
        self.alpha_fast_ema: final = ema_alpha_set(int((SEQ_LEN * perf_fast) * int(SEQ_LEN // offline_shift)))
        self.alpha_slow_ema: final = ema_alpha_set(int((SEQ_LEN * perf_slow) * int(SEQ_LEN // offline_shift)))

        self.model = model
        self.last_X_scaled: dict[str, np.ndarray] = {}
        self.last_X_unscaled: dict[str, np.ndarray] = {}

        self.per_alpha: final | float = per_alpha
        self.per_beta_max: final | float = per_beta_max
        self.per_beta_min: final | float = per_beta_min
        self.huber_pct: final | int = huber_pct

        self.scale_beta = perf_scale_beta
        self.trend_gamma = trend_gamma
        self.perf_scale_ema = None

        # this is used in the probability term to distribute sequences. this parameter boosts sentiment bias
        self.sent_gamma: final | float = sent_gamma

        # Performance parameters, experimental
        self.trend_floor = trend_floor
        self.trend_lambda = trend_lambda
        self.update_on_worse = True
        self.trend_alpha = trend_alpha

        self.trend_consecutive = 2
        self._trend_streak = 0
        self.warmup_updates = 10
        self.grad_steps = 0
        self.trend_scale_ema = None

        self.optimizer_args = optimizer_args
        self.optimizer, self.min_lr_abs, self.max_lr_abs, tot_steps = init_optimizer_from_args(optimizer_args)

        if per:
            replay_len = int(batch_size ** 2)
        else:
            replay_len = batch_size
        self.replay = deque(maxlen=replay_len)

        self.loss_hist = deque(maxlen=replay_len)
        if grad_clip:
            self.clip_hist = deque(maxlen=replay_len)
        self.pct_hist = deque(maxlen=replay_len)

        self.last_lstm_mae: Optional | float = None
        self.last_naive_mae: Optional | float = None
        #self.freeze_feature_blocks()
        self.model.summary()
        self.last_fetch_shift = -1
        print(f"ModelManager initialized with some params:\n"
              f"Performance alphas (slow/fast): {self.alpha_slow_ema}, {self.alpha_fast_ema}\n"
              f"Optimizer: {type(self.optimizer), self.optimizer.iterations.numpy()}/{tot_steps}\n"
              f"Min LR: {self.min_lr_abs}. Max LR: {self.max_lr_abs}\n"
              f"PER alpha: {self.per_alpha}\n")

    def freeze_feature_blocks(self, n_to_freeze=0):
        count = 0
        for layer in self.model.layers:
            if isinstance(layer, LSTM):
                layer.trainable = count >= n_to_freeze
                count += 1

    def do_replay(self, slice_shift: int) -> bool:
        """
        Define the timestep at which to append the next sequence to the replay buffer
        """
        if slice_shift % self.offline_shift == 0:
            return True

    def _update_trend_threshold(self, perf_trend: float) -> float:
        x = abs(float(perf_trend))
        if self.trend_scale_ema is None:
            self.trend_scale_ema = x
        else:
            a = self.trend_alpha
            self.trend_scale_ema = a * x + (1 - a) * self.trend_scale_ema
        thr = max(self.trend_floor, self.trend_lambda * self.trend_scale_ema)
        return float(min(thr, 0.9))

    def _update_trend_streak(self, perf_trend: float, thr: float = None) -> bool:

        if thr is None:
            thr = max(self.trend_floor, self.trend_lambda * (self.trend_scale_ema or 0.0))

        val = float(perf_trend)
        if self.update_on_worse:
            if val < -thr:
                self._trend_streak = self._trend_streak - 1 if self._trend_streak <= 0 else -1
            elif val > thr:
                self._trend_streak = 0
            else:
                self._trend_streak = int(np.sign(self._trend_streak)) if self._trend_streak != 0 else 0
            return abs(self._trend_streak) >= self.trend_consecutive
        else:
            if val > thr:
                self._trend_streak = self._trend_streak + 1 if self._trend_streak >= 0 else 1
            elif val < -thr:
                self._trend_streak = 0
            else:
                self._trend_streak = int(np.sign(self._trend_streak)) if self._trend_streak != 0 else 0
            return self._trend_streak >= self.trend_consecutive

    def do_update(self, slice_shift: int, fetch_shift: int, perf_trend: float, perf_delta: float,
                  n_buffer: int) -> bool:
        """
        n:th timestep to update the networks parameters on
        """

        if self.update_interval < 1 or fetch_shift < 1:
            return False
        t_seen = (fetch_shift * SEQ_LEN) + slice_shift
        #if self.grad_steps < self.warmup_updates:
        #    return (t_seen % self.update_interval) == 0
        if t_seen % self.update_interval == 0:
            return True
        #    self._update_trend_threshold(perf_trend)
        #    self._update_trend_streak(perf_trend)
        #    return False
        #thr = self._update_trend_threshold(perf_trend)
        #trigger = self._update_trend_streak(perf_trend, thr)
        #return trigger

    def randomize_shift(self) -> None:
        self.offline_shift = random.randint(3, int(SEQ_LEN * 0.5))

    def predict_close(self, symbol: str, X_scaled: np.ndarray, X_unscaled: np.ndarray, slice_shift: int,
                      fetch_shift: int,
                      max_days: int, lstm_mae: Optional, naive_mae: Optional, sigma_X_close: float, mu_X_close: float):
        """
        Called each timestep
        :param symbol:
        :param X_scaled: Varje tidstegs sekvens av längd SEQ_LEN
        :param X_unscaled:  Varje tidstegs sekvens
        :param slice_shift: tidsstegsvariabel
        :param sentiment_score:
        :param fetch_shift:
        :param max_days:
        :param update_shift:
        :return:
        """
        global TOTAL_UPDATES
        lr: Optional = None
        k: Optional = None
        if lstm_mae is not None and naive_mae is not None:
            self.last_lstm_mae = lstm_mae
            self.last_naive_mae = naive_mae
        symbol = symbol.upper()

        # arrays (seq_len, n_feat), the previous timestep (currently 30 minutes)
        last_scaled = self.last_X_scaled.get(symbol)
        last_unscaled = self.last_X_unscaled.get(symbol)

        if last_scaled is not None and last_scaled.shape[0] == SEQ_LEN:
            if not np.array_equal(last_scaled[-1], X_scaled[-1]):

                all_closes_raw = last_unscaled[:, CLOSE_IDX]
                mu_last = np.mean(all_closes_raw)
                sigma_last = np.std(all_closes_raw)
                new_close_raw = X_unscaled[self.const_label_index, CLOSE_IDX]
                #y_true_norm = (new_close_raw - mu_last) / sigma_last
                y_true_pct = (new_close_raw - float(all_closes_raw[-1])) / float(all_closes_raw[-1])
                y_true = np.array([[y_true_pct]], dtype=np.float32)

                # array (k, seq_len, n_feat)
                x_batch = last_scaled[np.newaxis, ...]
                if self.do_replay(slice_shift):
                    _, lr, k = self.online_update(symbol, x_batch, y_true,
                                                  new_close_raw, mu_last, sigma_last,
                                                  self.last_lstm_mae, self.last_naive_mae, slice_shift,
                                                  fetch_shift)

        # arrays (seq_len, n_feat)
        if self.do_replay(slice_shift):
            self.last_X_scaled[symbol] = X_scaled.copy()
            self.last_X_unscaled[symbol] = X_unscaled.copy()

        batch_x = X_scaled[np.newaxis, ...]
        y_vector = self.model.predict(batch_x, verbose=ZERO)

        all_y_for_stat = X_unscaled[:, CLOSE_IDX]
        y_pct: float = y_vector[0, 0]
        last_close = all_y_for_stat[-1]
        pred_close = last_close * (1.0 + y_pct)
        return y_pct, pred_close, lr, k

    def per_beta(self):
        """
        The idea here is to increase bias correction as the backtest goes on,
        but this also suffers from the ambigious step
        :return:
        """
        beta0 = self.per_beta_min
        target = self.per_beta_max
        horizon = self.beta_horizon_updates
        u = TOTAL_UPDATES / max(1, horizon)
        return float(min(target, beta0 + (target - beta0) * u))

    def sample_prioritized(self, k, alpha):
        """
        :param k:
        :param alpha:
        :return:
        """
        ps = np.array([p for (_, _, p, _) in self.replay],
                      dtype=np.float32)
        self.p_new = np.percentile(ps, self.priority_percentile)
        s_vals = np.array([_s for (_, _, _, _s) in self.replay], dtype=np.float32)

        mu = np.median(s_vals)
        mad = np.median(np.abs(s_vals - mu)) + 1e-8
        z = 1.4826 * (s_vals - mu) / mad
        zcap = self.zcap
        gamma = self.sent_gamma
        boost = np.exp(gamma * np.clip(np.abs(z), 0.0, zcap)).astype(np.float32)
        if self.sent_gamma == 0:
            assert boost.all() == 1
        probs = (ps ** alpha) * boost
        probs /= probs.sum()
        n = len(probs)
        uniform_probs = np.ones(n,
                                dtype=np.float32) / n
        uniform_eps = 0.1
        probs = (1 - uniform_eps) * probs + uniform_eps * uniform_probs
        probs /= probs.sum()
        #print(f"Update: {TOTAL_UPDATES}. S_gamma: {self.sent_gamma}. PER-Alpha: {self.per_alpha}. P_new: {self.p_new}.\n"
        #      f"{probs}")
        idx = np.random.choice(len(ps), size=k, replace=False, p=probs)
        batch = [self.replay[i][:2] for i in idx]
        beta = np.clip(self.per_beta(), 0.1, 1.0)
        is_weights = (1.0 / (n * probs[idx])) ** beta
        is_weights = is_weights / is_weights.max()
        is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        #print("idx:", idx)
        #print("probs:", probs.round(3))
        return batch, idx, is_weights

    def online_update(self, symbol: str, X_batch, y_true,
                      new_close_raw, mu_last, sigma_last,
                      lstm_mae: Optional, naive_mae: Optional, slice_shift: int, fetch_shift: int):
        """
        Note that self.offline_shift determines the shift between each (X_batch, y_true) as of now
        :param symbol:
        :param X_batch:
        :param y_true:
        :param new_close_raw:
        :param mu_last:
        :param sigma_last:
        :param lstm_mae:
        :param naive_mae:
        :param slice_shift:
        :param fetch_shift:
        :return:
        """
        global TOTAL_UPDATES
        symbol = symbol.upper()
        if X_batch.shape[1] != SEQ_LEN:
            print(f"Warning: Received seqlen {X_batch.shape[1]} for {symbol} in online_update.")
            return ZERO, ZERO, ZERO

        s_i = normalized_trend_slope(X_batch[0, :, 7])
        p_i = 1.0 if self.p_new is None else self.p_new
        self.replay.append((X_batch.squeeze(ZERO), y_true.squeeze(ZERO), p_i, s_i))
        buffer = list(self.replay)
        n_buffer = len(buffer)
        k = min(n_buffer, self.max_batch)
        alpha = self.per_alpha

        # Performance metric
        if lstm_mae is not None and naive_mae is not None:
            perf_delta = naive_mae - lstm_mae
        else:
            perf_delta = 0.0
        if self.fast_perf_ema is None:
            self.fast_perf_ema = perf_delta
            self.slow_perf_ema = perf_delta
        else:
            self.fast_perf_ema = self.alpha_fast_ema * perf_delta + (1 - self.alpha_fast_ema) * self.fast_perf_ema
            self.slow_perf_ema = self.alpha_slow_ema * perf_delta + (1 - self.alpha_slow_ema) * self.slow_perf_ema

        if self.perf_scale_ema is None:
            self.perf_scale_ema = abs(perf_delta)
        else:
            b = self.scale_beta
            self.perf_scale_ema = b * abs(perf_delta) + (1 - b) * self.perf_scale_ema
        perf_scale = max(self.perf_scale_ema, abs(self.slow_perf_ema), 1e-3)
        raw_trend = (self.fast_perf_ema - self.slow_perf_ema) / perf_scale
        perf_trend = float(np.tanh(self.trend_gamma * raw_trend))

        # avoiding dropout in probabilities
        training = self.do_update(slice_shift, fetch_shift, perf_trend, perf_delta, n_buffer)

        # prioritized sequences
        batch, sampled_idx, is_weight = self.sample_prioritized(k=k, alpha=alpha)
        batch_x = np.stack([x for x, y in batch], axis=0)
        batch_y = np.stack([y for x, y in batch], axis=0)

        # Standard GradientTape block from tensorflow docs
        with (tf.GradientTape() as tape):
            y_pred = self.model(batch_x, training=training)
            preds = tf.cast(y_pred, tf.float32)
            truth = tf.cast(batch_y, tf.float32)
            abs_errors = tf.abs(preds - truth).numpy()
            delta = float(np.clip(np.percentile(abs_errors, self.huber_pct), 1e-8, 1.0))
            self.pct_hist.append(delta)
            if len(self.pct_hist) > self.warmup_updates:
                delta = np.percentile(list(self.pct_hist), self.huber_pct)
            else:
                delta = 0.1
            loss_func = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
            #print("k:", int(tf.shape(preds)[0]))
            #print("preds.shape:", preds.shape, "truth.shape:", truth.shape)
            loss_vec = loss_func(truth, preds)
            tf.debugging.assert_rank(loss_vec, 1)
            tf.debugging.assert_equal(tf.shape(loss_vec)[0], tf.shape(preds)[0])
            loss = tf.reduce_mean(loss_vec * is_weight) if self.per else tf.reduce_mean(loss_vec)
            p_vec = loss_vec.numpy().astype(np.float32)
            p_floor = 1e-7

        grads = tape.gradient(loss, self.model.trainable_variables)
        #grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
        norm = tf.linalg.global_norm(grads).numpy()
        if self.grad_clip:
            self.clip_hist.append(norm)
            if len(self.clip_hist) > self.warmup_updates:
                clip = np.percentile(list(self.clip_hist), self.huber_pct)
            else:
                clip = None
            if clip is not None:
                grads_clipped, _ = tf.clip_by_global_norm(grads, clip)
                #grad_norms_c = [tf.norm(g).numpy() for g in grads_clipped if g is not None]
            else:
                grads_clipped = grads
                #grad_norms_c = grad_norms
        else:
            grads_clipped = grads
            #grad_norms_c = grad_norms

        # priority update loop
        for i, idx in enumerate(sampled_idx):
            x_i, y_i, _old_p, s_i = self.replay[idx]
            self.replay[idx] = (x_i, y_i, float(max(p_floor, p_vec[i])), s_i)

        if not training:
            return ZERO, ZERO, k

        self.optimizer.apply_gradients(zip(grads_clipped, self.model.trainable_variables))
        self.grad_steps += 1

        step = self.optimizer.iterations.numpy()
        current_lr = self.optimizer.learning_rate(step).numpy()
        print(f"Batch_size: {k}. [LR: {current_lr}, step: {step}]. Huber-delta: {delta}. {sampled_idx}")
        TOTAL_UPDATES += 1
        return ZERO, ZERO, k

    def save_model(self, path: str = "./files/online_model.keras"):
        """
        Save new weights
        :param path:
        :return:
        """
        self.model.save(path)
        print(f"Model saved to {path}")
