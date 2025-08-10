# /Backtest/model_manager.py
import itertools
from collections import deque

import tensorflow as tf
import numpy as np
from collections import defaultdict, deque
from scipy.stats import norm as sentnorm
from typing import Optional, final
from CONSTANTS import SEQ_LEN,WINDOW_LENGTH_DAY, CLOSE_IDX
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

"""


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
    alpha = 2 / (N + 1)
    return alpha


def get_sent_trend(n_buffer: int, sent_vectors_aggregated: list, sent_agg_sigma: float):
    if n_buffer < 2:
        return 0.0
    x = np.arange(len(sent_vectors_aggregated))
    slope = np.polyfit(x, sent_vectors_aggregated, 1)[0]
    trend = 0.5 + 0.5 * np.tanh(slope / sent_agg_sigma)
    return trend


def _current_lr(optimizer: tf.keras.optimizers.Optimizer):
    lr = optimizer.learning_rate
    if callable(lr):
        return float(lr(optimizer.iterations).numpy())
    return float(lr.numpy())


def _decay_steps_from_updates(updates_per_day: int, max_days: int) -> int:
    return max(1, int(updates_per_day) * int(max_days))


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
            ref_lr = 1e-5
        else:
            ref_lr = _current_lr(optimizer)

    max_lr_abs = float(ref_lr * lr_scale)
    min_lr_abs = float(max_lr_abs * min_lr_frac)

    if per:
        lr_var = tf.Variable(min_lr_abs, trainable=False, dtype=tf.float32)
        optimizer.learning_rate = lr_var
        return optimizer, min_lr_abs, max_lr_abs

    if updates_per_day is None:
        updates_per_day = max(1, 48 // max(1, update_interval))

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

    return optimizer, min_lr_abs, max_lr_abs


class ModelManager:
    """

    """

    def __init__(self, model, max_days: int, grad_clip: bool,
                 huber_pct: int, alpha_min: float, alpha_max: float,
                 batch_size: int, level: float, trend: float, w_perf: float, w_sent: float,
                 dsc_momentum: float, update_interval: int, offline_shift: int, per: bool, is_sent: bool,
                 perf_fast: float,
                 perf_slow: float):
        """

        :param model:
        :param max_days:
        :param grad_clip:
        :param huber_pct:
        :param alpha_min:
        :param alpha_max:
        :param batch_size: Max batch_size if per=True, else batch_size.
        :param level:
        :param trend:
        :param w_perf:
        :param w_sent:
        :param dsc_momentum: UNUSED
        :param update_interval: [Pass 0 to this to never update model. Otherwise, pass it a multiple of SEQ_LEN (it shouldn't require this,
                                but the math is bugged.)] Interval of timesteps to do an update on, with respect to sequence length. I.e.,
                                every update_interval of ((fetch_shift * SEQ_LEN) + slice_shift), do a parameter update.
        :param offline_shift: This parameter does not affect inference. It determines timesteps between sequences in the replay deque.
                        This parameter does however determine n timesteps online_update is called.
        :param per: Whether to sample "prioritized" sequences into each training batch.
        :param is_sent: Whether online_update should index the 8:th column of the incoming sequences for pseudo-sentiment statistics.
        """

        self.grad_clip: final | bool = grad_clip
        self.is_sent: final | bool = is_sent
        self.per: final | bool = per
        self.max_batch: final | int = batch_size
        self.update_interval: final | int = update_interval
        self.offline_shift: int = offline_shift
        self.beta_horizon_updates = int((SEQ_LEN / update_interval) * max_days) if (update_interval != 0) else 0
        print(self.beta_horizon_updates)
        assert level + trend == 1.0
        assert w_perf + w_sent == 1.0
        self.verbose: final | int = 1
        self.w_perf: final | float = w_perf
        self.w_sent: final | float = w_sent
        self.max_days: final | int = max_days
        self.fast_perf_ema: float | Optional = None
        self.slow_perf_ema: float | Optional = None

        # Note that N now needs to be thought of as timesteps
        self.alpha_fast_ema: final = ema_alpha_set(int((SEQ_LEN * perf_fast) * int(SEQ_LEN // offline_shift)))
        self.alpha_slow_ema: final = ema_alpha_set(int((SEQ_LEN * perf_slow) * int(SEQ_LEN // offline_shift)))

        self.model = model
        self.last_X_scaled: dict[str, np.ndarray] = {}
        self.last_X_unscaled: dict[str, np.ndarray] = {}

        self.alpha_min: final | float = alpha_min
        self.alpha_max: final | float = alpha_max
        self.huber_pct: final | int = huber_pct

        # highly experimental learning rate parameters
        self.k_ref: final | float = max(1, int(self.max_batch / 2))
        self.scale_beta = 0.05
        self.trend_gamma = 1.5
        self.perf_scale_ema = None
        self.lr_floor = 1e-8
        # Performance parameters, experimental
        self.trend_floor = 1e-8
        self.trend_lambda = 0.8
        self.update_on_worse = True
        self.trend_alpha = 0.05
        self.trend_consecutive = 2
        self._trend_streak = 0
        self.warmup_updates = 10
        self.grad_steps = 0
        self.trend_scale_ema = None

        self.optimizer, self.min_lr, self.max_lr = init_optimizer(model.optimizer,
                                                                  0.5,
                                                                  1,
                                                                  per,
                                                                  False,
                                                                  False,
                                                                  update_interval,
                                                                  offline_shift,
                                                                  max_days)

        if per:
            replay_len = int(batch_size * 2)
        else:
            replay_len = batch_size
        self.replay = deque(maxlen=replay_len)

        self.loss_hist = deque(maxlen=batch_size)
        if grad_clip:
            self.clip_hist = deque(maxlen=batch_size)
        self.pct_hist = deque(maxlen=batch_size)
        self.sentiment_history = deque(maxlen=batch_size)
        self.level: final | float = level
        self.trend: final | float = trend
        self.last_lstm_mae: Optional | float = None
        self.last_naive_mae: Optional | float = None
        #self.freeze_feature_blocks()
        self.model.summary()
        self.last_fetch_shift = -1
        print(f"ModelManager initialized with some params:\n"
              f"Performance alphas (slow/fast): {self.alpha_slow_ema}, {self.alpha_fast_ema}\n"
              f"Optimizer: {type(self.optimizer), self.optimizer.iterations.numpy()}\n"
              f"Min LR: {self.min_lr}. Max LR: {self.max_lr}.")

    def freeze_feature_blocks(self, n_to_freeze=1):
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
        if self.grad_steps < self.warmup_updates:
            return (t_seen % self.update_interval) == 0
        if t_seen % self.update_interval != 0:
            self._update_trend_threshold(perf_trend)
            self._update_trend_streak(perf_trend)
            return False
        thr = self._update_trend_threshold(perf_trend)
        trigger = self._update_trend_streak(perf_trend, thr)
        return trigger

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
        last_scaled = self.last_X_scaled.get(symbol)
        last_unscaled = self.last_X_unscaled.get(symbol)

        if last_scaled is not None and last_scaled.shape[0] == SEQ_LEN:
            if not np.array_equal(last_scaled[-1], X_scaled[-1]):

                all_closes_raw = last_unscaled[:, CLOSE_IDX]
                mu_last = np.mean(all_closes_raw)
                sigma_last = np.std(all_closes_raw)
                new_close_raw = X_unscaled[0, CLOSE_IDX]
                #y_true_norm = (new_close_raw - mu_last) / sigma_last
                y_true_pct = (new_close_raw - float(all_closes_raw[-1])) / float(all_closes_raw[-1])
                y_true = np.array([[y_true_pct]], dtype=np.float32)
                x_batch = last_scaled[np.newaxis, ...]
                if self.do_replay(slice_shift):
                    _, lr, k = self.online_update(symbol, x_batch, y_true,
                                                  new_close_raw, mu_last, sigma_last,
                                                  self.last_lstm_mae, self.last_naive_mae, slice_shift,
                                                  fetch_shift)

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
        beta0 = 0.1
        target = 1.0
        horizon = self.beta_horizon_updates
        u = TOTAL_UPDATES / max(1, horizon)
        return float(min(target, beta0 + (target - beta0) * u))

    def sample_prioritized(self, k, alpha):
        """
        :param k:
        :param alpha:
        :return:
        """
        ps = np.array([p + GLOBAL_EPS for (_, _, p) in self.replay],
                      dtype=np.float32)
        probs = ps ** alpha
        probs /= probs.sum()
        n = len(probs)
        uniform_probs = np.ones(n,
                                dtype=np.float32) / n
        uniform_eps = 0.1
        probs = (1 - uniform_eps) * probs + uniform_eps * uniform_probs
        probs /= probs.sum()
        idx = np.random.choice(len(ps), size=k, replace=False, p=probs)
        batch = [self.replay[i][:2] for i in idx]
        beta = np.clip(self.per_beta(), 0.1, 1.0)
        is_weights = (1 / n * probs[idx]) ** beta
        is_weights = is_weights / is_weights.max()
        is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        #print("p:", ps.round(3))
        #print("probs:", probs.round(3))
        #print("idx:", idx)
        #print(f"a: {alpha}")
        return batch, idx, is_weights

    def online_update(self, symbol: str, X_batch, y_true,
                      new_close_raw, mu_last, sigma_last,
                      lstm_mae: Optional, naive_mae: Optional, slice_shift: int, fetch_shift: int):
        global TOTAL_UPDATES
        symbol = symbol.upper()
        if X_batch.shape[1] != SEQ_LEN:
            print(f"Warning: Received seqlen {X_batch.shape[1]} for {symbol} in online_update.")
            return ZERO, ZERO, ZERO

        self.replay.append((X_batch.squeeze(ZERO), y_true.squeeze(ZERO), 1.0))
        buffer = list(self.replay)
        n_buffer = len(buffer)

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
        scale = max(self.perf_scale_ema, abs(self.slow_perf_ema), 1e-3)
        raw_trend = (self.fast_perf_ema - self.slow_perf_ema) / scale
        perf_trend = float(np.tanh(self.trend_gamma * raw_trend))
        perf_score = 0.5 + 0.5 * perf_trend

        # Sentiment pseudo logic
        if self.is_sent:
            sentiment_series = X_batch[0, :, 7]
            tau = 8
            idx = np.arange(len(sentiment_series))[::-1]
            weights = np.exp(-idx / tau)
            weights /= weights.sum()
            sentiment_score = float(np.dot(weights, sentiment_series))
            self.sentiment_history.append(sentiment_score)
            sent_hist = list(self.sentiment_history)
            std_sent = np.std(sent_hist) + GLOBAL_EPS
            rank = sum(1 for x in sent_hist if x < sentiment_score)
            sent_level = rank / len(sent_hist)
            sent_trend = get_sent_trend(n_buffer, sent_hist, std_sent)
            base_norm = self.level * sent_level + self.trend * sent_trend
        else:
            base_norm = 0.0
        base_norm = np.clip(base_norm, 0.0, 1.0)

        # Effective batch size
        prop = (1.0 - base_norm)
        k = int(np.ceil(np.clip(prop, 0.10, 1.0) * min(n_buffer, self.max_batch)))
        k = np.clip(a=k, a_min=int(self.max_batch / 4), a_max=min(n_buffer, self.max_batch))
        if not self.per:
            # Resampling within batch should not matter for a stateless, so I let the method execute (otherwise replay population is greater than batch population)
            k = n_buffer

        k_gain = np.sqrt(max(1, k) / self.k_ref)
        new_lr = (self.min_lr + (self.max_lr - self.min_lr) * perf_score) * k_gain
        new_lr = float(np.clip(new_lr, self.min_lr, self.max_lr))
        # Adaptive LR exclusive with prioritized sampling
        if self.per:
            self.optimizer.learning_rate.assign(new_lr)

        alpha_min, alpha_max = self.alpha_min, self.alpha_max
        k_alpha_fac = (k - 1) / (self.replay.maxlen - 1)
        ratio = k_alpha_fac * 2
        alpha = alpha_min + (alpha_max - alpha_min) * (1 - ratio)
        batch, sampled_idx, is_weight = self.sample_prioritized(k=k, alpha=alpha)
        batch_x = np.stack([x for x, y in batch], axis=0)
        batch_y = np.stack([y for x, y in batch], axis=0)

        training = self.do_update(slice_shift, fetch_shift, perf_trend, perf_delta, n_buffer)
        # Standard GradientTape block from tensorflow docs
        with (tf.GradientTape() as tape):
            y_pred = self.model(batch_x, training=training)
            preds = y_pred[:, ZERO]
            truth = batch_y[:, ZERO]
            abs_errors = tf.abs(preds - truth).numpy()
            delta = float(np.clip(np.percentile(abs_errors, self.huber_pct), 1e-8, 1.0))
            self.pct_hist.append(delta)
            if len(self.pct_hist) > self.warmup_updates:
                delta = np.percentile(list(self.pct_hist), self.huber_pct)
            else:
                delta = 0.1
            loss_func = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
            loss = loss_func(truth, preds)
            loss = tf.reduce_mean(loss * is_weight) if self.per else tf.reduce_mean(loss)
        # loss_scalar = float(loss.numpy())

        grads = tape.gradient(loss, self.model.trainable_variables)
        #grad_norms = [tf.norm(g).numpy() for g in grads if g is not None]
        norm = tf.linalg.global_norm(grads).numpy()
        if self.grad_clip:
            self.clip_hist.append(norm)
            if len(self.clip_hist) > 2:
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
        for i, idx in enumerate(sampled_idx):
            x_i, y_i, _old_p = self.replay[idx]
            self.replay[idx] = (x_i, y_i, float(abs_errors[i]))

        if not training:
            return ZERO, new_lr, k

        self.optimizer.apply_gradients(zip(grads_clipped, self.model.trainable_variables))
        self.grad_steps += 1
        if self.per:
            current_lr = self.optimizer.learning_rate.numpy()
            print(
                f"Batch_size: {k}. LR: [O:{current_lr}, A: {new_lr}]. Huber-delta: {delta}.")
        else:
            step = self.optimizer.iterations.numpy()
            current_lr = self.optimizer.learning_rate(step).numpy()
            print(
                f"Batch_size: {k}. [LR: {current_lr}, step: {step}]. Huber-delta: {delta}.")
        TOTAL_UPDATES += 1
        return ZERO, new_lr, k

    def save_model(self, path: str = "./files/online_model.keras"):
        """
        Save new weights
        :param path:
        :return:
        """
        self.model.save(path)
        print(f"Model saved to {path}")

