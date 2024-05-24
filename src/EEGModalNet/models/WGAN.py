import torch
from keras import layers, ops
import keras


class WGAN_GP(keras.Model):
    def __init__(self, time_dim=100, feature_dim=2, latent_dim=64):
        super().__init__()
        self.time = time_dim
        self.feature = feature_dim
        self.input_shape = (time_dim, feature_dim)
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = keras.Sequential([
            keras.Input(shape=(self.latent_dim,)),
            layers.Dense(128, activation='relu'),
            # layers.BatchNormalization(),
            layers.Dense(256, activation='relu'),
            # layers.BatchNormalization(),
            layers.Dense(self.time * self.feature),
            layers.Reshape(self.input_shape)
        ], name='generator')

        # self.generator = keras.Sequential([
        #     keras.Input(shape=(self.latent_dim,)),
        #     layers.Dense(128, activation='relu'),
        #     layers.Dense(256, activation='relu'),
        #     layers.Reshape((256 // self.feature, self.feature)),
        #     layers.UpSampling1D(size=2),
        #     layers.Conv1D(self.feature, 3, padding='same'),
        #     layers.LeakyReLU(),
        #     # layers.UpSampling1D(size=2),
        #     # layers.Conv1D(self.feature, 3, padding='same'),
        #     # layers.LeakyReLU(),
        #     # layers.UpSampling1D(size=2),
        #     layers.Conv1D(self.feature, 3, padding='same'),
        # ], name='generator')

        self.discriminator = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Flatten(),
            layers.Dense(self.time * self.feature, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        # self.discriminator = keras.Sequential([
        #     keras.Input(shape=self.input_shape),
        #     layers.Conv1D(self.feature, 3, padding='same'),
        #     layers.LeakyReLU(),
        #     layers.Conv1D(self.feature, 3, padding='same'),
        #     layers.MaxPooling1D(pool_size=2),
        #     layers.Conv1D(self.feature, 3, padding='same'),
        #     layers.LeakyReLU(),
        #     layers.MaxPooling1D(pool_size=2),
        #     layers.Dense(64 * self.feature, activation='relu'),
        #     layers.Flatten(),
        #     layers.Dense(64, activation='relu'),
        #     layers.Dense(1, activation='sigmoid')
        # ], name='discriminator')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def call(self, x):
        return self.discriminator(x)

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

        prob_interpolated = self.discriminator(interpolated)

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

        means = real_data.mean(axis=1).mean(axis=1)
        stds = real_data.std(axis=1).mean(axis=1)
        noise = torch.zeros((batch_size, self.latent_dim))
        for i in range(batch_size):
            noise[i] = keras.random.normal((self.latent_dim,), mean=means[i], stddev=stds[i])
        # noise = keras.random.normal((batch_size, self.latent_dim),
        #                             mean=0, stddev=1)

        # train discriminator
        fake_data = self.generator(noise).detach()
        real_pred = self.discriminator(real_data)
        fake_pred = self.discriminator(fake_data)
        gp = self.gradient_penalty(real_data, fake_data.detach())
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        noise = torch.zeros((batch_size, self.latent_dim))
        for i in range(batch_size):
            noise[i] = keras.random.normal((self.latent_dim,), mean=means[i], stddev=stds[i])
        # noise = keras.random.normal((batch_size, self.latent_dim),
        #                             mean=0, stddev=1)

        self.zero_grad()
        fake_pred = self.discriminator(self.generator(noise))
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
