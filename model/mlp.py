import torch


class SimpleMLP(torch.nn.Module):

    def __init__(self, in_channels, out_channels=3, num_mlp_layers=12, dim_mlp=512, output_activation=torch.nn.Sigmoid()):
        super().__init__()
        self.output_channels = out_channels
        self.output_activation = output_activation
        layers = [torch.nn.Linear(in_channels, dim_mlp)]
        for i in range(num_mlp_layers - 2):
            layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(dim_mlp, dim_mlp))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(dim_mlp, out_channels))
        self.mlp = torch.nn.Sequential(*layers)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, feat_xyz):
        out = self.mlp(feat_xyz)
        out = self.output_activation(out)
        return out
