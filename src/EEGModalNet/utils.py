import keras
from tqdm import tqdm


class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, save_freq=200):
        super().__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if (epoch + 1) % self.save_freq == 0:
            self.model.save(f'{self.filepath}_epoch_{epoch+1}.keras')
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
        self.step_losses = {'g_loss': [], 'd_loss': []}

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.step_losses['g_loss'].append(logs.get('g_loss'))
        self.step_losses['d_loss'].append(logs.get('d_loss'))
