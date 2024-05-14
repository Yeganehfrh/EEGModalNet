import keras
from keras import ops, layers
from keras.layers import SpectralNormalization
import torch


class SimpleGAN(keras.Model):
    def __init__(self, time_dim=100, feature_dim=2, latent_dim=64, **kwargs):
        super().__init__(**kwargs)
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
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.time * self.feature),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.discriminator = keras.Sequential([
            keras.Input(shape=self.input_shape),
            SpectralNormalization(layers.Conv1D(64, 3, activation='relu', padding='same')),
            layers.Flatten(),
            layers.Dense(self.time * 64, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def compile(self, loss, d_optimizer, g_optimizer):
        super().compile(loss=loss)
        self.loss_fn = loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        # self.additional_metrics = metrics

    def call(self, x):
        return self.generator(x)

    def train_step(self, real_data):
        x_real = real_data
        batch_size = ops.shape(x_real)[0]

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    # seed=self.seed_generator
                                    )

        x_fake = self.generator(noise)
        x = ops.concatenate([x_real, x_fake], axis=0)

        # 0=real, 1=fake
        y = ops.concatenate(
            [ops.zeros((batch_size, 1)),
             ops.ones((batch_size, 1))], axis=0
        )

        # trick that adds noise to labels
        y = y + 0.05 * keras.random.uniform(ops.shape(y))

        # train discriminator
        self.zero_grad()
        y_pred = self.discriminator(x, training=True)
        d_loss = self.loss_fn(y, y_pred)  # TODO use compute_loss?
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    # seed=self.seed_generator
                                    )

        y_misleading = ops.zeros((batch_size, 1))  # saying all are real (0)

        self.zero_grad()
        y_pred = self.discriminator(self.generator(noise))
        g_loss = self.loss_fn(y_misleading, y_pred)
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


class ConditionalGAN(keras.Model):
    def __init__(self, time_dim=100, feature_dim=2, latent_dim=64, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.time = time_dim
        self.feature = feature_dim
        self.input_shape = (time_dim, feature_dim)
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        self.generator = keras.Sequential([
            keras.Input(shape=(self.latent_dim + self.num_classes,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.time * self.feature),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.discriminator = keras.Sequential([
            keras.Input(shape=(self.input_shape[0], self.input_shape[1] + self.num_classes)),
            layers.Flatten(),
            layers.Dense(self.time * self.feature, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def compile(self, loss, d_optimizer, g_optimizer):
        super().compile(loss=loss)
        self.loss_fn = loss
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def call(self, x):
        return self.discriminator(x)

    def train_step(self, real_data, labels):
        x_real = real_data
        batch_size = ops.shape(x_real)[0]

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    # seed=self.seed_generator
                                    )

        x_fake = self.generator(ops.concatenate([noise, labels], axis=1))
        x = ops.concatenate([x_real, x_fake], axis=0)

        # 0=real, 1=fake
        y = ops.concatenate(
            [ops.zeros((batch_size, 1)),
             ops.ones((batch_size, 1))], axis=0
        )

        # trick that adds noise to labels
        y = y + 0.05 * keras.random.uniform(ops.shape(y))

        # train discriminator
        self.zero_grad()
        y_pred = self.discriminator(ops.concatenate([x, labels], axis=1), training=True)
        d_loss = self.loss_fn(y, y_pred)  # TODO use compute_loss?
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1,
                                    # seed=self.seed_generator
                                    )

        y_misleading = ops.zeros((batch_size, 1))  # saying all are real (0)

        self.zero_grad()
        y_pred = self.discriminator(ops.concatenate([self.generator(ops.concatenate([noise, labels], axis=1)), labels], axis=1))
        g_loss = self.loss_fn(y_misleading, y_pred)
        grads = g_loss.backward()
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
