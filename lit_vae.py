import lightning as L
import torch.nn as nn
import torch
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder


class VAE(L.LightningModule):
    def __init__(
        self, enc_out_dim=512, latent_dim=256, input_height=32, num_classes=10
    ):
        super().__init__()
        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        # self.encoder.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False,
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # one hot encoding to embed labels
        self.label_emb = nn.Embedding(num_classes, embedding_dim=enc_out_dim)

    def forward(self, x):
        x, y = x
        # x = x.to(self.device)
        x = x.to(self.device)
        y = y.to(self.device)

        x_encode = self.encoder(x)
        # print(f"Input shape: {x.shape}")  # Debug print
        x_encoded_with_label = x_encode * self.label_emb(y)

        mu, log_var = self.fc_mu(x_encoded_with_label), self.fc_var(
            x_encoded_with_label
        )

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decode the image
        x_hat = self.decoder(z)

        return x_hat, mu, std, z, x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = log_qzx - log_pz
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x_hat, mu, std, z, x = self(batch)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        loss = elbo.mean()

        self.log_dict(
            {
                "elbo": loss,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
                "reconstruction": recon_loss.mean(),
            }
        )

        return loss
