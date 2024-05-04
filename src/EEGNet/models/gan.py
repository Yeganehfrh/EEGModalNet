import keras
from keras import ops, layers


class Generator(keras.Model):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.generator = keras.Sequential([
            keras.layers.InputLayer((latent_dim,)),
            layers.Dense(out_channels, activation='relu')
        ],
            name="generator")

    def call(self, inputs):
        inputs = self.generator(inputs)
        # unflatten the inputs
        inputs = ops.reshape(inputs, (-1, 28, 28, 1))  # TODO: change this to the correct shape
        return inputs


class Discriminator(keras.Model):
    def __init__(self, in_channels):
        super().__init__()
        self.discriminator = keras.Sequential([
            keras.layers.InputLayer((in_channels,)),
            keras.layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ],
            name="discriminator")

    def call(self, inputs):
        return self.discriminator(inputs)


class GAN(keras.Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, 256 * 12)
        self.discriminator = Discriminator(256 * 12)
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
    
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        labels_ = labels[:, None]
        labels_ = ops.repeat(
            labels_, repeats=[image_size * image_size]
        )
        image_one_hot_labels = ops.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = ops.shape(real_images)[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = ops.concatenate(
            [generated_images, image_one_hot_labels], -1
        )
        real_image_and_labels = ops.concatenate([real_images, image_one_hot_labels], -1)
        combined_images = ops.concatenate(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = ops.concatenate(
            [ops.ones((batch_size, 1)), ops.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )
        random_vector_labels = ops.concatenate(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = ops.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = ops.concatenate(
                [fake_images, image_one_hot_labels], -1
            )
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }



    # def train_step(self, data):
    #     x, y = data

    #     # Train discriminator
    #     with ops.GradientTape() as tape:
    #         fake_y = self.generator(x, training=True)
    #         fake_y = ops.stop_gradient(fake_y)
    #         combined_y = ops.concatenate([y, fake_y], axis=0)
    #         labels = ops.concatenate([ops.ones((y.shape[0], 1)), ops.zeros((fake_y.shape[0], 1))], axis=0)
    #         d_loss = self.loss_fn(labels, combined_y)
    #     d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
    #     self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

    #     # Train generator
    #     with ops.GradientTape() as tape:
    #         fake_y = self.generator(x, training=True)
    #         labels = ops.ones((x.shape[0], 1))
    #         g_loss = self.loss_fn(labels, fake_y)
    #     g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
    #     self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

    #     return {"d_loss": d_loss, "g_loss": g_loss}

    # def test_step(self, data):
    #     x, y = data
    #     fake_y = self.generator(x, training=False)
    #     labels = ops.ones((x.shape[0], 1))
    #     g_loss = self.loss_fn(labels, fake_y)
    #     return {"g_loss": g_loss}

    # def generate(self, inputs):
    #     return self.generator(inputs, training=False)

    # def discriminate(self, inputs):
    #     return self.discriminator(inputs, training=False)