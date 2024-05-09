import os
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import keras
from src.EEGModalNet.models.gan import SimpleGAN
from src.EEGModalNet.data.mock_temporal_data import MockTemporalData
from src.EEGModalNet.utils import ProgressBarCallback
from tqdm.auto import tqdm


def run(max_epochs=100_000, n_features=1, latent_dim=64):

    reusable_pbar = tqdm(total=max_epochs, unit='epoch', leave=False, dynamic_ncols=True)

    # data
    data = MockTemporalData(n_samples=20, n_features=n_features, n_timepoints=100)
    x, _ = data()

    model = SimpleGAN(time_dim=100, feature_dim=n_features, latent_dim=latent_dim)
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  d_optimizer=keras.optimizers.Adam(0.001),
                  g_optimizer=keras.optimizers.Adam(0.001))

    history = model.fit(
        x,
        epochs=max_epochs,
        verbose=0, # type: ignore
        shuffle=True,
        callbacks=[
            keras.callbacks.ModelCheckpoint('tmp/keras_models/simple_gan_v1.model.keras',
                                            monitor='d_loss', mode='min'),
            keras.callbacks.EarlyStopping(monitor='d_loss', mode='min', patience=max_epochs // 5),
            keras.callbacks.CSVLogger('tmp/keras_logs/simple_gan_v1.csv'),
            ProgressBarCallback(n_epochs=max_epochs, n_runs=1, run_index=0, reusable_pbar=reusable_pbar),
        ]
    )


if __name__ == '__main__':
    run()
