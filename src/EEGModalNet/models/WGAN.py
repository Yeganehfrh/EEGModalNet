import torch
from keras import layers
import keras
from .common import SubjectLayers_v2, SubjectLayers, convBlock, ChannelMerger, ResidualBlock
from ..preprocessing.spectral_regularization import spectral_regularization_loss


class Critic(keras.Model):
    def __init__(self, time_dim, feature_dim, n_subjects, use_sublayer, use_channel_merger, *args, **kwargs):
        super(Critic, self).__init__()

        self.input_shape = (time_dim, feature_dim)
        self.use_sublayer = use_sublayer
        negative_slope = 0.1
        kernel_initializer = keras.initializers.Orthogonal(gain=1.0)

        if use_sublayer:
            self.sub_layer = SubjectLayers(feature_dim, feature_dim, n_subjects, init_id=True)  # TODO: check out the input and output channels when we include more channels

        if use_channel_merger:
            self.pos_emb = ChannelMerger(
                chout=feature_dim * 8, pos_dim=128, n_subjects=n_subjects, per_subject=True,  # TODO: pos_dim has a temporary value
            )
            self.input_shape = (time_dim, feature_dim * 8)

        self.model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            ResidualBlock(8 * feature_dim, kernel_initializer=kernel_initializer, 5, activation='relu'),  # TODO: update kernel size argument
            # TransformerEncoder(feature_dim, 4, 2, 8, 0.2),
            layers.SpectralNormalization(layers.Conv1D(8, 15, padding='same', activation='relu', name='conv3', kernel_initializer=kernel_initializer)),
            layers.AveragePooling1D(2, name='downsampling1'),
            layers.SpectralNormalization(layers.Conv1D(8, 9, padding='same', activation='relu', name='conv3', kernel_initializer=kernel_initializer)),
            layers.AveragePooling1D(2, name='downsampling2'),
            layers.SpectralNormalization(layers.Conv1D(8, 7, padding='same', activation='relu', name='conv4', kernel_initializer=kernel_initializer)),
            layers.AveragePooling1D(2, name='downsampling3'),
            # layers.SpectralNormalization(layers.Conv1D(8, 7, padding='same', activation='relu', name='conv4')),
            # layers.AveragePooling1D(2, name='downsampling4'),
            layers.Flatten(name='dis_flatten'),
            layers.SpectralNormalization(layers.Dense(512, name='dis_dense1', kernel_initializer=kernel_initializer)),
            layers.LeakyReLU(negative_slope=negative_slope),
            # layers.Dropout(dropout_rate),
            layers.SpectralNormalization(layers.Dense(128, name='dis_dense2', kernel_initializer=kernel_initializer)),
            layers.LeakyReLU(negative_slope=negative_slope),
            # layers.Dropout(dropout_rate),
            layers.SpectralNormalization(layers.Dense(32, name='dis_dense3', kernel_initializer=kernel_initializer)),
            layers.LeakyReLU(negative_slope=negative_slope),
            # layers.Dropout(dropout_rate),
            layers.SpectralNormalization(layers.Dense(8, name='dis_dense4', kernel_initializer=kernel_initializer)),
            layers.LeakyReLU(negative_slope=negative_slope),
            # layers.Dropout(dropout_rate),
            layers.SpectralNormalization(layers.Dense(1, name='dis_dense5', dtype='float32', kernel_initializer=kernel_initializer)),
        ], name='critic')

        self.built = True

    def call(self, x, sub_labels, positions):
        if hasattr(self, 'sub_layer'):
            x = self.sub_layer(x, sub_labels)
        if hasattr(self, 'pos_emb'):
            x = self.pos_emb(x, sub_labels, positions)
        out = self.model(x)
        return out


class Generator(keras.Model):
    def __init__(self, time_dim, feature_dim, latent_dim, use_sublayer, num_classes, emb_dim, kerner_initializer,
                 n_subjects, use_channel_merger, interpolation, *args, **kwargs):
        super(Generator, self).__init__()
        self.negative_slope = 0.2
        self.input_shape = (time_dim, feature_dim)
        self.use_sublayer = use_sublayer
        self.latent_dim = latent_dim

        if use_sublayer:
            self.sub_layer = SubjectLayers(feature_dim, feature_dim, n_subjects, init_id=True)
            # self.sub_layer = SubjectLayers_v2(num_classes, emb_dim)

        if use_channel_merger:
            self.pos_emb = ChannelMerger(
                chout=feature_dim, pos_dim=32, n_subjects=n_subjects  # TODO: pos_dim has a temporary value + chout might need to be updated
            )

        self.model = keras.Sequential([
            keras.Input(shape=((latent_dim,))),
            layers.Dense(128 * 1, kernel_initializer=kerner_initializer, name='gen_layer1'),
            layers.LeakyReLU(negative_slope=self.negative_slope, name='gen_layer2'),
            layers.Dense(256 * 1, kernel_initializer=kerner_initializer, name='gen_layer3'),
            layers.LeakyReLU(negative_slope=self.negative_slope, name='gen_layer4'),
            layers.Reshape((8, 32), name='gen_layer9'),
            *convBlock(filters=6 * [8 * feature_dim],
                       kernel_sizes=[19, 17, 15, 9, 7, 5],
                       upsampling=6 * [1],
                       stride=1,
                       padding='same',
                       interpolation=interpolation,
                       negative_slope=0.2,
                       kernel_initializer=kerner_initializer,
                       batch_norm=True),
            layers.Conv1D(feature_dim, 3, padding='same', name='last_conv_lyr', kernel_initializer=kerner_initializer)
        ], name='generator')

        self.built = True

    def call(self, noise, sub_labels, positions):
        x = self.model(noise)
        if hasattr(self, 'pos_emb'):
            x = self.pos_emb(x, sub_labels, positions)
        if hasattr(self, 'sub_layer'):
            x = self.sub_layer(x, sub_labels)  # TODO: this layer can be used before or after data generation
            if keras.mixed_precision.global_policy().name == 'mixed_float16':
                x = x.float()  # make sure the output is in float32 in mixed precision mode
        return x


@keras.saving.register_keras_serializable()
class WGAN_GP(keras.Model):
    def __init__(self,
                 time_dim=100, feature_dim=2, latent_dim=64, n_subjects=1,
                 use_sublayer_generator=False, use_sublayer_critic=False,
                 emb_dim=20, kerner_initializer='glorot_uniform',
                 use_channel_merger_g=False,
                 use_channel_merger_c=False,
                 interpolation='bilinear',
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
        # self.critic_updates = critic_updates

        self.generator = Generator(time_dim=time_dim,
                                   feature_dim=feature_dim,
                                   latent_dim=latent_dim,
                                   use_sublayer=use_sublayer_generator,
                                   num_classes=n_subjects,
                                   emb_dim=emb_dim,
                                   kerner_initializer=kerner_initializer,
                                   n_subjects=n_subjects,
                                   use_channel_merger=use_channel_merger_g,
                                   interpolation=interpolation)

        self.critic = Critic(time_dim=time_dim,
                             feature_dim=feature_dim,
                             n_subjects=n_subjects,
                             emb_dim=emb_dim,
                             use_sublayer=use_sublayer_critic,
                             use_channel_merger=use_channel_merger_c,)

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

    def gradient_penalty(self, real_data, fake_data, sub, pos):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, device=real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic(interpolated, sub, pos)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)  # TODO: it was view before changed to reshape because of an error
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, data):
        # real_data, sub, pos = data[0], data[1], data[2]  # TODO: find a better way to handle the subject device
        real_data, sub, pos = data['x'], data['sub'], data['pos']

        batch_size = real_data.size(0)
        mean = real_data.mean()
        std = real_data.std()

        # train critic
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std)
        fake_data = self.generator(noise, sub, pos).detach()
        real_pred = self.critic(real_data, sub, pos)
        fake_pred = self.critic(fake_data, sub, pos)  # TODO: should we use the same sub and pos for fake data?
        gp = self.gradient_penalty(real_data, fake_data.detach(), sub, pos)
        self.zero_grad()
        # spectral_regularization_loss_value = spectral_regularization_loss(real_data, fake_data, lambda_match=1 / 10e9, include_smooth=False)
        d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
        d_loss.backward()

        grads = [v.value.grad for v in self.critic.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # Monitor gradient norms
        gradient_norms = []
        for p in self.critic.parameters():
            if p.grad is not None:
                gradient_norms.append(p.grad.norm().item())

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim), mean=mean, stddev=std, dtype=real_data.dtype)

        self.zero_grad()
        random_sub = torch.randint(0, sub.max().item(), (batch_size, 1), device=real_data.device)  # TODO: change it back to real labels if necessary
        x_gen = self.generator(noise, random_sub, pos)  # TODO: consider using random positions
        fake_pred = self.critic(x_gen, sub, pos)
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
            '_real_pred': real_pred.mean().item(),
            '_fake_pred': fake_pred.mean().item(),
            '_gp': gp.item(),
        }
