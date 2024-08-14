import torch
from keras import layers
import keras


class Generator(keras.Model):
    def __init__(self, latent_dim, num_classes, time_series_length):
        super(Generator, self).__init__()

        self.init_size = time_series_length // 4
        self.emb_layer_g = torch.nn.Embedding(num_classes, num_classes)

        self.model = keras.Sequential([
            keras.Input(shape=(latent_dim + num_classes,)),
            layers.Dense(256),
            layers.LeakyReLU(negative_slope=0.3),
            layers.Dense(256 * 2),
            layers.LeakyReLU(negative_slope=0.3),
            layers.Reshape((256, 2)),
            layers.BatchNormalization(),
            layers.UpSampling1D(2),
            layers.Conv1D(4, 3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.UpSampling1D(2),
            layers.Conv1D(8, 3, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv1D(1, 3, padding="same"),
            layers.Activation('tanh')
        ], name='generator')

        self.built = True

    def call(self, noise, labels):
        gen_input = torch.cat((self.emb_layer_g(labels).squeeze(), noise), -1)
        out = self.model(gen_input)
        return out.permute(0, 2, 1)


class Critic(keras.Model):
    def __init__(self, num_classes, features, time_series_length):
        super(Critic, self).__init__()

        self.emb_layer_c = torch.nn.Embedding(num_classes, num_classes)

        self.model = keras.Sequential([
            keras.Input(shape=(features + num_classes, time_series_length, )),
            layers.Conv1D(64, 3, strides=2, padding="same", name='conv1'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv1D(128, 3, strides=2, padding="same", name='conv2'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv1D(128, 3, strides=2, padding="same", name='conv3'),
            layers.BatchNormalization(),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv1D(1, 3, strides=2, padding="same", name='conv4'),
            layers.Flatten(),
            layers.Dense(1, name='dense')
        ], name='critic')
        self.built = True

    def call(self, time_series, labels):
        label_embedding = self.emb_layer_c(labels).repeat(1, time_series.size(2), 1).permute(0, 2, 1)
        d_in = torch.cat((time_series, label_embedding), 1)
        out = self.model(d_in)
        return out


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

        self.generator = Generator(self.latent_dim, self.num_classes, self.time)
        self.critic = Critic(self.num_classes, self.feature, self.time)
        # self.build = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def get_config(self):
        config = super().get_config()
        return config

    # def call(self, x, labels):
    #     return self.critic(x, labels)

    def compile(self, d_optimizer, g_optimizer, gradient_penalty_weight):
        super().compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gradient_penalty_weight = gradient_penalty_weight

    def gradient_penalty(self, real_data, fake_data, labels):
        epsilon = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic(interpolated, labels)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(real_data.size(0), -1)  # TODO: it was view before changed to reshape because of error
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def train_step(self, data):
        real_data, sub_labels = data['x'], data['sub']
        batch_size = real_data.size(0)

        mean = real_data.mean()
        std = real_data.std()
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)

        # train discriminator
        fake_data = self.generator(noise, sub_labels).detach()
        real_pred = self.critic(real_data, sub_labels)  # TODO: maybe we should use generated_labels here
        fake_pred = self.critic(fake_data, sub_labels)

        gp = self.gradient_penalty(real_data, fake_data, sub_labels)
        self.zero_grad()
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()  # TODO: check if retain_graph=True is necessary
        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # # Clip weights of discriminator  # TODO: check if this is necessary
        # for p in self.critic.parameters():
        #     p.data.clamp_(-clip_value, clip_value)

        # train generator
        # self.g_optimizer.zero_grad()  # TODO: check if this is necessary
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)
        self.zero_grad()
        gen_labels = torch.randint(0, self.num_classes, (batch_size, 1)).to(real_data.device)
        fake_data = self.generator(noise, gen_labels)

        fake_pred = self.critic(fake_data, gen_labels)

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
