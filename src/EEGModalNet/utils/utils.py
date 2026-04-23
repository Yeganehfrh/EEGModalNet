import os

import keras
import torch
from tqdm import tqdm


def get_resume_state_path(filepath):
    return f'{filepath}.resume.pt'


def _optimizer_variable_tensors(optimizer):
    tensors = []
    for variable in optimizer.variables:
        value = variable.value if hasattr(variable, 'value') else variable
        if isinstance(value, torch.Tensor):
            tensors.append(value.detach().cpu().clone())
        else:
            tensors.append(torch.as_tensor(keras.ops.convert_to_numpy(value)).cpu().clone())
    return tensors


def _restore_optimizer_variables(optimizer, values):
    optimizer_variables = list(optimizer.variables)
    if len(optimizer_variables) != len(values):
        raise ValueError(
            f'Optimizer variable count mismatch: expected {len(optimizer_variables)}, got {len(values)}.'
        )

    for variable, saved_value in zip(optimizer_variables, values):
        target = variable.value if hasattr(variable, 'value') else variable
        if isinstance(target, torch.Tensor):
            target.copy_(saved_value.to(device=target.device, dtype=target.dtype))
        else:
            variable.assign(keras.ops.convert_to_numpy(saved_value))


def load_training_state(model, checkpoint_path):
    resume_state_path = get_resume_state_path(checkpoint_path)
    if not os.path.exists(resume_state_path):
        return 0, 'missing'

    state = torch.load(resume_state_path, map_location='cpu')
    saved_epoch = int(state.get('epoch', 0))
    saved_global_step = int(state.get('global_step', 0))

    if hasattr(model.d_optimizer, 'build'):
        model.d_optimizer.build(model.critic.trainable_weights)
    if hasattr(model.g_optimizer, 'build'):
        model.g_optimizer.build(model.generator.trainable_weights)

    try:
        _restore_optimizer_variables(model.d_optimizer, state['d_optimizer_variables'])
        _restore_optimizer_variables(model.g_optimizer, state['g_optimizer_variables'])
    except ValueError as exc:
        model.global_step = saved_global_step
        print(
            f'>>>> Optimizer state mismatch in {resume_state_path}: {exc}. '
            'Continuing from checkpoint weights and saved epoch/global_step only.'
        )
        return saved_epoch, 'weights_only'

    model.global_step = saved_global_step
    return saved_epoch, 'exact'


class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=200, save_training_state=False):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.save_training_state = save_training_state

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = f'{self.filepath}_epoch_{epoch+1}.model.keras'
            self.model.save(checkpoint_path)
            if self.save_training_state:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'global_step': int(getattr(self.model, 'global_step', 0)),
                        'd_optimizer_variables': _optimizer_variable_tensors(self.model.d_optimizer),
                        'g_optimizer_variables': _optimizer_variable_tensors(self.model.g_optimizer),
                    },
                    get_resume_state_path(checkpoint_path),
                )
            print(f"Checkpoint saved at epoch {epoch+1}")


class ProgressBarCallback(keras.callbacks.Callback):
    def __init__(self, n_epochs=None,
                 n_runs=None,
                 run_index=None,
                 reusable_pbar: tqdm = None):

        self.n_epochs = n_epochs
        self.pbar: tqdm = reusable_pbar
        if self.pbar is None:
            self.pbar = tqdm(
                total=n_epochs,
                unit='epoch',
                dynamic_ncols=True,
                leave=False)
        self.pbar.total = n_epochs
        self.pbar.set_description(f'run {run_index:02}/{n_runs:02}')

    def on_train_begin(self, logs=None):
        self.pbar.reset()

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.set_postfix(logs)
        self.pbar.update(epoch - self.pbar.n + 1)

    def on_train_end(self, logs=None):
        self.pbar.reset()


class StepLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.step_stats = {'g_loss': [], 'd_loss': [], 'critic_grad_norm': [], 'gp': []}

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.step_stats['g_loss'].append(logs.get('g_loss'))
        self.step_stats['d_loss'].append(logs.get('d_loss'))
        self.step_stats['critic_grad_norm'].append(logs.get('critic_grad_norm'))
        self.step_stats['gp'].append(logs.get('_gp'))


@keras.saving.register_keras_serializable()
class BalancedAccuracy(keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = keras.ops.cast(keras.ops.reshape(y_true, (-1,)), "float32")
        y_pred = keras.ops.cast(keras.ops.reshape(y_pred, (-1,)), "float32")
        y_pred = keras.ops.cast(y_pred >= self.threshold, "float32")

        tp = keras.ops.sum(y_pred * y_true)
        tn = keras.ops.sum((1.0 - y_pred) * (1.0 - y_true))
        fp = keras.ops.sum(y_pred * (1.0 - y_true))
        fn = keras.ops.sum((1.0 - y_pred) * y_true)

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        sensitivity = self.true_positives / (
            self.true_positives + self.false_negatives + keras.backend.epsilon()
        )
        specificity = self.true_negatives / (
            self.true_negatives + self.false_positives + keras.backend.epsilon()
        )
        return (sensitivity + specificity) / 2.0

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold})
        return config
