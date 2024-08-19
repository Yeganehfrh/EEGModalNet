import torch
from keras import layers, ops
import keras
from src.EEGModalNet.models.common import SubjectLayers_v2, ChannelMerger


class Critic(keras.Model):
    def __init__(self, num_classes, emb_dim=20, use_sublayers=False, *args, **kwargs):
        super(Critic, self).__init__()

        if use_sublayers:
            self.subject_layers = SubjectLayers_v2(num_classes, emb_dim)

        self.model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            layers.Conv1D(1, 3, padding='same', activation='relu', name='conv1'),
            layers.Flatten(name='dis_flatten'),
            layers.Dense(256, activation='relu', name='dis_dense1'),
            layers.Dense(128, activation='relu', name='dis_dense2'),
            layers.Dense(64, activation='relu', name='dis_dense3'),
            layers.Dense(1, name='dis_dense4')
        ], name='critic')
        self.built = True

    def call(self, time_series, labels):
        if hasattr(self, 'subject_layers'):
            x = self.subject_layers(time_series, labels)
        out = self.model(x)
        return out

@keras.saving.register_keras_serializable()
class WGAN_GP(keras.Model):
    def __init__(self,
                 time_dim=100, feature_dim=2, latent_dim=64, n_subjects=1, use_sublayers=False, emb_dim=20,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = time_dim
        self.feature = feature_dim
        self.input_shape = (time_dim, feature_dim)
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(128),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Reshape((256 // 1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.5),
            layers.UpSampling1D(size=2),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Conv1D(1, 3, padding='same'),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.critic = Critic(n_subjects, emb_dim=emb_dim, use_sublayers=use_sublayers)

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
        super().compile(run_eagerly=True)
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
        gradients = gradients.reshape(batch_size, -1)  # TODO: it was view before changed to reshape because of error
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, data):
        real_data, sub = data['x'], data['sub']
        batch_size = real_data.size(0)

        mean = real_data.mean()
        std = real_data.std()
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        # train critic
        fake_data = self.generator(noise).detach()
        real_pred = self.critic(real_data, sub)
        fake_pred = self.critic(fake_data, sub)
        gp = self.gradient_penalty(real_data, fake_data.detach())
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()
        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        self.zero_grad()
        x_gen = self.generator(noise)
        random_sub = torch.randint(0, sub.max().item(), (batch_size, 1)).to(real_data.device)  # TODO: change it back to real labels if necessary
        fake_pred = self.critic(x_gen, random_sub)
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
