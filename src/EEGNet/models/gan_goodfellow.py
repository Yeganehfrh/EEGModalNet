import keras
from keras import layers, ops  # Input, Dense, Reshape, Flatten, Dropout
from keras import Sequential, Model
from keras import optimizers  # Adam

import matplotlib.pyplot as plt

import numpy as np


class GAN():
    def __init__(self):
        self.time = 128
        self.channels = 61
        self.batch_size = 3
        self.data_shape = (self.time, self.channels)
        self.latent_dim = 100

        optimizer = optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = keras.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(0.0002, 0.5))  # TODO

    def build_generator(self, out_channels=784):

        model = Sequential()
        model.add(layers.Dense(128 * 61, input_dim=self.latent_dim, name='gen_dense'))
        model.add(layers.LeakyReLU(alpha=0.2, name='gen_leakyrelu'))
        model.add(layers.Reshape(self.data_shape, name='gen_reshape'))

        model.summary()

        noise = keras.Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(layers.Flatten(input_shape=self.data_shape, name='dis_flatten'))
        model.add(layers.Dense(128 * 61, name='dis_dense'))
        model.add(layers.LeakyReLU(alpha=0.2, name='dis_leakyrelu'))
        model.add(layers.Dense(1, activation='sigmoid', name='dis_dense2'))
        model.summary()

        img = keras.Input(shape=self.data_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, input, epochs, batch_size=1, sample_interval=50):

        # Load the dataset
        X_train = input

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1)).astype(np.float32)
        fake = np.zeros((batch_size, 1)).astype(np.float32)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(np.float32)

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)).astype(np.float32)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print(d_loss[0], 100 * d_loss[1], g_loss)
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # # If at save interval => save generated image samples
            # if epoch % sample_interval == 0:
            #     self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


# if __name__ == '__main__':
#     gan = GAN()
#     gan.train(data, epochs=30000, batch_size=32, sample_interval=200)
