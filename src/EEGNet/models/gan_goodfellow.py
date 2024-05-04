import keras
from keras import layers
from keras import Sequential, Model
from keras import optimizers  # Adam
from tqdm import tqdm

import numpy as np


class GAN():
    def __init__(self, time, features):
        self.time = time
        self.features = features
        self.data_shape = (self.time, self.features)
        self.latent_dim = 100

        optimizer = optimizers.Adam(0.001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = keras.Input(shape=(self.latent_dim,))
        fake_x = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        fake_y = self.discriminator(fake_x)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, fake_y)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.001))  # TODO

    def build_generator(self):

        model = Sequential()
        model.add(keras.Input(shape=(self.latent_dim,)))
        model.add(layers.Dense(self.time * self.features, input_dim=self.latent_dim,
                               activation=keras.layers.LeakyReLU(negative_slope=0.2)))
        model.add(layers.Dense(self.time * self.features))
        model.add(layers.Reshape(self.data_shape))

        return model

    def build_discriminator(self):

        model = Sequential()
        model.add(keras.Input(shape=self.data_shape))
        model.add(layers.Flatten(input_shape=self.data_shape))
        model.add(layers.Dense(self.time * self.features,
                               activation=keras.layers.LeakyReLU(negative_slope=0.2)))
        model.add(layers.Dense(8))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def train(self, input, epochs, batch_size=1, sample_interval=50):

        # Load the dataset
        X_train = input

        # Adversarial ground truths
        valid = np.ones((batch_size, 1)).astype(np.float32)
        fake = np.zeros((batch_size, 1)).astype(np.float32)

        for epoch in tqdm(range(epochs)):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            x = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(np.float32)

            # Generate a batch of new images
            x_fake = self.generator.predict(noise, verbose=0)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(x, valid)
            d_loss_fake = self.discriminator.train_on_batch(x_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(np.float32)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
