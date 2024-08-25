import torch
from keras import layers, ops
import keras
from src.EEGModalNet.models.common import SubjectLayers_v2, convBlock


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='same', activation=activation)
        self.conv2 = layers.Conv1D(filters // 2, kernel_size, padding='same')
        self.activation = layers.Activation(activation)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = layers.add([x, inputs])  # Add the input (shortcut connection)
        x = self.activation(x)
        return x


class Critic(keras.Model):
    def __init__(self, time_dime, feature_dim, num_classes, emb_dim, use_sublayer, *args, **kwargs):
        super(Critic, self).__init__()

        self.input_shape = (time_dime, feature_dim)
        self.use_sublayer = use_sublayer

        if use_sublayer:
            self.sub_layer = SubjectLayers_v2(num_classes, emb_dim)

        self.model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            ResidualBlock(8, 5, activation='relu'),
            # ResidualBlock(4, 5, activation='relu'), 
            # layers.Conv1D(8, 5, padding='same', activation='relu', name='conv1'),
            # layers.Conv1D(4, 5, padding='same', activation='relu', name='conv2'),
            layers.Conv1D(1, 5, padding='same', activation='relu', name='conv3'),
            layers.Flatten(name='dis_flatten'),
            layers.Dense(512, activation='relu', name='dis_dense1'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', name='dis_dense2'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', name='dis_dense3'),
            layers.Dropout(0.2),
            layers.Dense(8, activation='relu', name='dis_dense4'),
            layers.Dropout(0.2),
            layers.Dense(1, name='dis_dense5')
        ], name='critic')
        self.built = True

    def call(self, x, labels):
        if hasattr(self, 'sub_layer'):
            print('using sublayer in critic')
            x = self.sub_layer(x, labels)
        out = self.model(x)
        return out


class Generator(keras.Model):
    def __init__(self, time_dim, feature_dim, latent_dim, use_sublayer, num_classes, emb_dim, *args, **kwargs):
        super(Generator, self).__init__()
        self.negative_slope = 0.2
        self.input_shape = (time_dim, feature_dim)
        self.use_sublayer = use_sublayer
        self.latent_dim = latent_dim

        if use_sublayer:
            self.sub_layer = SubjectLayers_v2(num_classes, emb_dim)

        self.model = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(128),
            layers.LeakyReLU(negative_slope=self.negative_slope),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=self.negative_slope),
            layers.Reshape((256, 1)),
            *convBlock([1, 1], [3, 5], [1, 1], 1, 'same', 0.2, True),
            layers.Conv1D(1, 7, padding='same', name='last_conv_lyr'),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.built = True

    def call(self, noise, labels):
        x = self.model(noise)
        if hasattr(self, 'sub_layer'):
            x = self.sub_layer(x, labels)  # TODO: this layer can be used before or after data generation
        return x


@keras.saving.register_keras_serializable()
class WGAN_GP(keras.Model):
    def __init__(self,
                 time_dim=100, feature_dim=2, latent_dim=64, n_subjects=1,
                 use_sublayer_generator=False, use_sublayer_critic=False,
                 emb_dim=20,
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

        self.generator = Generator(time_dim, feature_dim, latent_dim, use_sublayer_generator, n_subjects, emb_dim)
        self.critic = Critic(self.time, self.feature, n_subjects, emb_dim=emb_dim, use_sublayer=use_sublayer_critic)

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

    def gradient_penalty(self, real_data, fake_data, sub):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1).to(real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic(interpolated, sub)

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

        # train critic
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)
        fake_data = self.generator(noise, sub).detach()  # TODO: consider using random labels
        real_pred = self.critic(real_data, sub)
        fake_pred = self.critic(fake_data, sub)
        gp = self.gradient_penalty(real_data, fake_data.detach(), sub)
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()

        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # monitor gradient norms to ensure a stable training
        gradient_norms = []
        for p in self.critic.parameters():
            if p.grad is not None:
                gradient_norms.append(p.grad.norm().item())

        # # Train the critic more frequently than the generator
        # critic_updates = 5  # Set the number of critic updates per generator update

        # for _ in range(critic_updates):
        #     noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)
        #     fake_data = self.generator(noise, sub).detach()
        #     real_pred = self.critic(real_data, sub)
        #     fake_pred = self.critic(fake_data, sub)
        #     gp = self.gradient_penalty(real_data, fake_data.detach(), sub)
        #     self.zero_grad()
        #     d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        #     d_loss.backward()

        #     grads = [v.value.grad for v in self.critic.trainable_weights]
        #     with torch.no_grad():
        #         self.d_optimizer.apply(grads, self.critic.trainable_weights)

        #     # Monitor gradient norms
        #     gradient_norms = []
        #     for p in self.critic.parameters():
        #         if p.grad is not None:
        #             gradient_norms.append(p.grad.norm().item())

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        self.zero_grad()
        random_sub = torch.randint(0, sub.max().item(), (batch_size, 1)).to(real_data.device)  # TODO: change it back to real labels if necessary
        x_gen = self.generator(noise, random_sub)
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
            'critic_grad_norm': sum(gradient_norms) / len(gradient_norms),
        }
