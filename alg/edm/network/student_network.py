from .base_network import *

class StudentNetwork(BaseNetwork):
    def __init__(self,
        in_dim  : int,
        out_dim : int,
        width   : int,
    ):
        super(StudentNetwork, self).__init__()

        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.width   = width

        self.layers = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, out_dim),
        )

    def forward(self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(x)
