import torch
from keras import layers, ops
import keras


class ConditionalWGAN(keras.Model):
    def __init__(self, time_dim=100, feature_dim=2, latent_dim=64, num_classes=2):
        super().__init__()
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
            layers.Dense(256, activation='relu'),
            layers.Dense(self.time * self.feature),
            layers.Reshape(self.input_shape)
        ], name='generator')

        self.discriminator = keras.Sequential([
            keras.Input(shape=(self.input_shape[0], self.input_shape[1] + self.num_classes)),
            layers.Flatten(),
            layers.Dense(self.time * self.feature, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='softmax')
        ], name='discriminator')

        self.classifier = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(2, activation='softmax')], name='classifier')

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def call(self, x):
        return self.discriminator(x)

    def compile(self, d_optimizer, g_optimizer, loss_fn, gradient_penalty_weight):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
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

    def train_step(self, data):
        real_data, real_labels = data['x'], data['y']
        batch_size = real_data.size(0)

        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1)

        # update the real_labels dimension to be able to concatenate with the data
        real_labels_reshaped = real_labels.unsqueeze(1).repeat(1, self.time, 1)

        # train discriminator
        fake_data = self.generator(torch.cat((noise, real_labels), dim=1)).detach()
        real_data_labels = torch.cat((real_data, real_labels_reshaped), dim=-1)
        fake_data_labels = torch.cat((fake_data, real_labels_reshaped), dim=-1)
        combined_images = ops.concatenate(
            [fake_data_labels, real_data_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )
        # gp = self.gradient_penalty(real_data_labels), fake_data_labels.detach())
        self.zero_grad()
        # label_loss = keras.losses.binary_crossentropy(real_labels, real_pred[:, 1:])
        # d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight + label_loss
        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim),
                                    mean=0, stddev=1)
        
        noise_labels = ops.concatenate([noise, real_labels], axis=1)
        misleading_labels = ops.zeros((batch_size, 1))

        self.zero_grad()
        fake_data = self.generator(noise_labels)
        fake_data_labels = ops.concatenate([fake_data, real_labels_reshaped], axis=-1)
        predictions = self.discriminator(fake_data_labels)
        g_loss = self.loss_fn(misleading_labels, predictions)
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
