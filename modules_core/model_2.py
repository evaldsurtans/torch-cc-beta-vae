import torch
import torchvision.models

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=4, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=8, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),
            torch.nn.AvgPool2d(kernel_size=4, stride=2, padding=0),

            torch.nn.Conv2d(in_channels=16, out_channels=self.args.embedding_size, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=self.args.embedding_size, num_groups=8)
        )

        self.encoder_mu = torch.nn.Linear(in_features=self.args.embedding_size, out_features=self.args.embedding_size)
        self.encoder_sigma = torch.nn.Linear(in_features=self.args.embedding_size, out_features=self.args.embedding_size)

        self.decoder = torch.nn.Sequential(
            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=self.args.embedding_size, out_channels=16, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=16, num_groups=8),


            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=16, out_channels=8, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),

            torch.nn.UpsamplingBilinear2d(scale_factor=2),
            torch.nn.Conv2d(in_channels=8, out_channels=4, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),

            torch.nn.AdaptiveAvgPool2d(output_size=(28, 28)),
            torch.nn.Conv2d(in_channels=4, out_channels=1, padding=1, stride=1, kernel_size=3, bias=False),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(normalized_shape=[1, 28, 28]),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder.forward(x)

        out_flat = h.view(x.size(0), -1)

        z_mu = self.encoder_mu.forward(out_flat)
        z_sigma = self.encoder_sigma.forward(out_flat)

        if self.args.gamma > 0:
            # re-parameterization trick
            eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size()).to(self.args.device) # -3.0..3.0
            z = z_mu + eps * z_sigma # (B, 32)
        else:
            z = z_mu

        # (B, 32, 1, 1)
        y_prim = self.decoder.forward(z.view(-1, self.args.embedding_size, 1, 1))
        return z, z_mu, z_sigma, y_prim

    def encode_z(self, x):
        out = self.encoder(x)

        out_flat = out.view(x.size(0), -1)

        z_sigma = self.encoder_sigma.forward(out_flat)
        z_mu = self.encoder_mu.forward(out_flat)

        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size())
        z = z_mu + eps * z_sigma

        return z

    def decode_z(self, z):
        z_2d = z.view(z.size(0), -1, 1, 1)
        y_prim = self.decoder(z_2d)
        return y_prim


