import torch
from keras import layers, ops
import keras
from src.EEGModalNet.models.common import SubjectLayers


class WGAN_GP(keras.Model):
    def __init__(self, time_dim=100, feature_dim=2, latent_dim=64, n_subjects=1, use_sublayers=False):
        super().__init__()
        self.time = time_dim
        self.feature = feature_dim
        self.input_shape = (time_dim, feature_dim)
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        if use_sublayers:
            self.subject_layers = SubjectLayers(self.time, self.time, n_subjects)

        self.generator = keras.Sequential([
            keras.Input(shape=(self.latent_dim,)),
            layers.Dense(128),
            layers.LeakyReLU(0.3),
            layers.Dense(256),
            layers.LeakyReLU(0.3),
            layers.Reshape((256 // self.feature, self.feature)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.3),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.critic = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Flatten(name='dis_flatten'),
            layers.Dense(self.time * self.feature, activation='relu', name='dis_dense1'),
            layers.Dense(64, activation='relu', name='dis_dense3'),
            layers.Dense(1, name='dis_dense4')
        ], name='critic')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

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

    def train_step(self, data):
        real_data, sub = data['x'], data['sub']
        batch_size = real_data.size(0)

        means = real_data.mean(axis=1).mean(axis=1)
        stds = real_data.std(axis=1).mean(axis=1)
        noise = torch.zeros((batch_size, self.latent_dim))
        for i in range(batch_size):
            noise[i] = keras.random.normal((self.latent_dim,), mean=means[i], stddev=stds[i])
        # noise = keras.random.normal((batch_size, self.latent_dim),
        #                             mean=0, stddev=1)
        if hasattr(self, 'subject_layers'):
            real_data = self.subject_layers(real_data, sub)

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

        # train generator
        noise = torch.zeros((batch_size, self.latent_dim))
        for i in range(batch_size):
            noise[i] = keras.random.normal((self.latent_dim,), mean=means[i], stddev=stds[i])
        # noise = keras.random.normal((batch_size, self.latent_dim),
        #                             mean=0, stddev=1)

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
