import torch
import torchvision.models

class Model(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # densenet161, shufflenet
        self.encoder = torchvision.models.resnet18(pretrained=False)
        self.encoder.conv1 = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.fc_mu = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.encoder.fc.out_features,
                out_features=self.args.embedding_size
            ),
            torch.nn.Tanh() # maybe nothing
        )

        self.fc_sigma = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.encoder.fc.out_features,
                out_features=self.args.embedding_size),
            torch.nn.Sigmoid() # ReLU
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.args.embedding_size,
                out_channels=16,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.ReLU(),

            torch.nn.GroupNorm(num_channels=16, num_groups=8),
            torch.nn.UpsamplingBilinear2d(scale_factor=4), # (B,16,4,4)

            torch.nn.Conv2d(
                in_channels=16,
                out_channels=8,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=8, num_groups=4),
            torch.nn.UpsamplingBilinear2d(scale_factor=2), # (B,8,8,8)

            torch.nn.Conv2d(
                in_channels=8,
                out_channels=4,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.GroupNorm(num_channels=4, num_groups=4),
            torch.nn.UpsamplingBilinear2d(scale_factor=2), # (B,4,16,16)


            torch.nn.Conv2d(
                in_channels=4,
                out_channels=1,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(normalized_shape=[1, 16, 16]),
            torch.nn.UpsamplingBilinear2d(size=(28, 28)), # (B,1,28,28)

            torch.nn.Conv2d(
                in_channels=1,
                out_channels=1,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder.forward(x)

        z_sigma = self.fc_sigma.forward(h)
        z_mu = self.fc_mu.forward(h)

        # re-parameterization trick
        eps = torch.normal(mean=0.0, std=1.0, size=z_mu.size()).to(self.args.device) # -3.0..3.0
        z = z_mu + eps * z_sigma # (B, 32)

        # (B, 32, 1, 1)
        y_prim = self.decoder.forward(z.view(-1, self.args.embedding_size, 1, 1))
        return z, z_mu, z_sigma, y_prim



