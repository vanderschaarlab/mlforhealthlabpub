from .__head__ import *

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def forward(self,
        x : torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
