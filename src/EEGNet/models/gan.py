import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.EEGNet.models.autoencoders import convNet, mlp  # noqa
import torchmetrics.functional as tmf


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features=latent_dim, out_features=12 * 256),
                                    nn.LeakyReLU())
        self.layer2 = nn.Unflatten(1, (256, 12))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Flatten(),
                                    nn.Linear(in_features=12 * 256, out_features=256),
                                    nn.LeakyReLU(),
                                    nn.Linear(in_features=256, out_features=1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        return x


class GAN(pl.LightningModule):

    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.automatic_optimization = False
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()

    def forward(self, x):
        return self.discriminator(x)

    def generator_step(self, x_real):
        batch_size = x_real.shape[0]
        # Sample noise
        noise = torch.randn(batch_size, self.latent_dim)

        # Generate data
        x_fake = self.generator(noise)

        x = torch.cat((x_real, x_fake))
        # TODO get only batch of x

        # Classify generated data
        # using the discriminator
        y_pred = self.discriminator(x)

        y = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)))

        # Backprop loss
        g_loss = nn.BCELoss()(y, y_pred.squeeze())

        return g_loss

    def discriminator_step(self, x_real):
        batch_size = x_real.shape[0]
        # Real images
        y_real = torch.squeeze(self.discriminator(x_real))
        loss_real = nn.BCELoss()(y_real, torch.zeros(batch_size))

        # Fake images
        noise = torch.randn(batch_size, self.latent_dim)
        x_fake = self.generator(noise)
        y_fake = self.discriminator(x_fake).squeeze()
        loss_fake = nn.BCELoss()(y_fake, torch.ones(x_fake.shape[0]))
        acc = tmf.accuracy(torch.cat((y_real, y_fake)),
                           torch.cat((torch.zeros(batch_size), torch.ones(batch_size))),
                           task='binary',
                           num_classes=2)
        self.log('accuracy', acc, prog_bar=True)

        return (loss_real + loss_fake) / 2

    def training_step(self, batch):
        X, _, _, _ = batch
        opt1, opt2 = self.optimizers()

        # train generator
        opt1.zero_grad()
        g_loss = self.generator_step(X)
        self.manual_backward(g_loss)
        opt1.step()
        self.log('g_loss', g_loss, prog_bar=True)

        # train discriminator
        opt2.zero_grad()
        d_loss = self.discriminator_step(X)
        self.manual_backward(d_loss)
        opt2.step()
        self.log('d_loss', d_loss, prog_bar=True)

        return {'g_loss': g_loss, 'd_loss': d_loss}

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002)
        return [g_optimizer, d_optimizer], []
