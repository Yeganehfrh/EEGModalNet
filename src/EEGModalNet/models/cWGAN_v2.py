import torch
from keras import layers, ops
import keras


@keras.saving.register_keras_serializable()
class cWGAN_GP(keras.Model):
    def __init__(self,
                 time_dim=100, feature_dim=2, latent_dim=64, n_subjects=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = time_dim
        self.feature = feature_dim
        self.input_shape = (time_dim, feature_dim)
        self.latent_dim = latent_dim
        self.num_classes = n_subjects
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.emb_layer = torch.nn.Embedding(self.num_classes, self.num_classes)

        self.generator = keras.Sequential([
            keras.Input(shape=(latent_dim + self.num_classes,)),
            layers.Dense(128),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Dense(128),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Reshape((128 // 1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.5),
            layers.UpSampling1D(size=2),
            layers.Conv1D(self.feature, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.5),
            layers.UpSampling1D(size=2),
            layers.Conv1D(1, 3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.5),
            layers.Conv1D(1, 3, padding='same'),
            layers.Reshape(self.input_shape)
        ], name='generator')

        # self.conv_blocks = nn.Sequential(
        #     nn.BatchNorm1d(128),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv1d(128, 128, 3, stride=1, padding=1),
        #     nn.BatchNorm1d(128, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv1d(128, 64, 3, stride=1, padding=1),
        #     nn.BatchNorm1d(64, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv1d(64, 1, 3, stride=1, padding=1),
        #     nn.Tanh(),
        # )

        # self.critic = keras.Sequential([
        #     keras.Input(shape=(self.input_shape[0], self.input_shape[1] + self.num_classes)),
        #     layers.Conv1D(1, 3, padding='same', activation='relu', name='conv1'),
        #     layers.Flatten(name='dis_flatten'),
        #     layers.Dense(256, activation='relu', name='dis_dense1'),
        #     layers.Dense(128, activation='relu', name='dis_dense2'),
        #     layers.Dense(64, activation='relu', name='dis_dense3'),
        #     layers.Dense(1, name='dis_dense4')
        # ], name='critic')

        self.critic = keras.Sequential([
            keras.Input(shape=(self.time + self.num_classes,)),
            layers.Dense(512, name='dis_dense1'),
            layers.Dropout(0.2),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(256, name='dis_dense2'),
            layers.Dropout(0.2),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(128, name='dis_dense3'),
            layers.Dropout(0.2),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1, name='dis_dense4')
        ], name='critic')

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
        epsilon = torch.rand(batch_size, 1).to(real_data.device)
        interpolated = epsilon * real_data.squeeze() + (1 - epsilon) * fake_data.squeeze()
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
        real_data, sub_labels = data['x'], data['sub']
        batch_size = real_data.size(0)

        mean = real_data.mean()
        std = real_data.std()
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        # update the real_labels dimension to be able to concatenate with the data
        # sub_labels_reshaped = sub_labels.unsqueeze(1).repeat(1, self.time, 1)
        sub_labels = self.emb_layer(sub_labels).squeeze()

        # train discriminator
        fake_data = self.generator(torch.cat((noise, sub_labels), dim=1)).detach()
        real_data_labels = torch.cat((real_data.squeeze(), sub_labels), dim=-1)
        fake_data_labels = torch.cat((fake_data.squeeze(), sub_labels), dim=-1)
        real_pred = self.critic(real_data_labels)
        fake_pred = self.critic(fake_data_labels)

        gp = self.gradient_penalty(real_data_labels, fake_data_labels.detach())
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward(retain_graph=True)  # TODO: check if retain_graph=True is necessary
        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        noise_labels = ops.concatenate([noise, sub_labels], axis=1)
        self.zero_grad()
        fake_data = self.generator(noise_labels)

        fake_data_labels = torch.cat([fake_data.squeeze(), sub_labels], dim=-1)
        fake_pred = self.critic(fake_data_labels)

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
