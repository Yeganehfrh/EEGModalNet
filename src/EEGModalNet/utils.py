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
