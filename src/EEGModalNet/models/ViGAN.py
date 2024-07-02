import torch
from keras import Sequential
import keras
from keras.layers import Input, Dense, Reshape, UpSampling1D, Conv1D, BatchNormalization, ReLU, UpSampling2D, Conv2D, TimeDistributed
from keras.layers import Flatten, Conv1D, GlobalAveragePooling1D, Dropout


@keras.saving.register_keras_serializable()
class ViT_GAN(keras.Model):
    def __init__(self, frames, height, width, channels, latent_dim=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = build_generator(latent_dim, frames, height, width, channels)

        self.critic = Sequential([
            Input(shape=(frames, height, width, 3)),
            # Apply Conv2D to each frame independently
            TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2))),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2))),
            TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=(2, 2))),
            # Flatten each frame's feature maps
            TimeDistributed(Flatten()),
            # Now the shape should be (batch_size, num_frames, flattened_features)
            # Apply Conv1D across the frames
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalAveragePooling1D(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        return self.critic(x)

    def compile(self, d_optimizer, g_optimizer, gradient_penalty_weight):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gradient_penalty_weight = gradient_penalty_weight

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic(interpolated)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_data):
        batch_size = real_data.size(0)

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1)

        # train critic
        fake_data = self.generator(noise).detach()
        real_pred = self.critic(real_data)
        fake_pred = self.critic(fake_data)
        gp = self.gradient_penalty(real_data, fake_data.detach())
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()
        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1)

        self.zero_grad()
        fake_pred = self.critic(self.generator(noise))
        g_loss = -fake_pred.mean()
        g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            'd_loss': self.d_loss_tracker.result(),
            'g_loss': self.g_loss_tracker.result(),
        }


def build_generator(latent_dim, num_frames, height, width):
    model = Sequential()

    model.add(Input(shape=(latent_dim,)))

    # Fully connected layer to reshape input into a small tensor
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Reshape((num_frames // 8, 256//16)))

    # Upsampling in the temporal dimension
    model.add(UpSampling1D(size=2))
    model.add(Conv1D(32, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(UpSampling1D(size=2))
    model.add(Conv1D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(height // 8 * width // 8 * 3, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Reshape((num_frames, height // 8, width // 8, 3)))

    # Upsampling in the spatial dimensions
    model.add(TimeDistributed(UpSampling2D(size=(2))))
    model.add(TimeDistributed(Conv2D(4, kernel_size=(3, 3), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(ReLU()))

    model.add(TimeDistributed(UpSampling2D(size=(2, 2))))
    model.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(ReLU()))

    model.add(TimeDistributed(UpSampling2D(size=(2, 2))))
    model.add(TimeDistributed(Conv2D(16, kernel_size=(3, 3), padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(ReLU()))

    model.add(TimeDistributed(Conv2D(3, kernel_size=(3, 3), padding='same')))

    return model
