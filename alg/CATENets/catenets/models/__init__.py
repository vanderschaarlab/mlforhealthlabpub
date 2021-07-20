from typing import Any

from catenets.models.disentangled_nets import SNet3
from catenets.models.pseudo_outcome_nets import DRNet, PseudoOutcomeNet, PWNet, RANet
from catenets.models.representation_nets import SNet1, SNet2, TARNet
from catenets.models.rnet import RNet
from catenets.models.snet import SNet
from catenets.models.tnet import TNet
from catenets.models.xnet import XNet

SNET1_NAME = "SNet1"
T_NAME = "TNet"
SNET2_NAME = "SNet2"
PSEUDOOUT_NAME = "PseudoOutcomeNet"
SNET3_NAME = "SNet3"
SNET_NAME = "SNet"
XNET_NAME = "XNet"
RNET_NAME = "RNet"
DRNET_NAME = "DRNet"
PWNET_NAME = "PWNet"
RANET_NAME = "RANet"
TARNET_NAME = "TARNet"

ALL_MODELS = [
    T_NAME,
    SNET1_NAME,
    SNET2_NAME,
    SNET3_NAME,
    SNET_NAME,
    PSEUDOOUT_NAME,
    RNET_NAME,
    XNET_NAME,
    DRNET_NAME,
    PWNET_NAME,
    RANET_NAME,
    TARNET_NAME,
]
MODEL_DICT = {
    T_NAME: TNet,
    SNET1_NAME: SNet1,
    SNET2_NAME: SNet2,
    SNET3_NAME: SNet3,
    SNET_NAME: SNet,
    PSEUDOOUT_NAME: PseudoOutcomeNet,
    RNET_NAME: RNet,
    XNET_NAME: XNet,
    DRNET_NAME: DRNet,
    PWNET_NAME: PWNet,
    RANET_NAME: RANet,
    TARNET_NAME: TARNet,
}

__all__ = [
    T_NAME,
    SNET1_NAME,
    SNET2_NAME,
    SNET3_NAME,
    SNET_NAME,
    PSEUDOOUT_NAME,
    RNET_NAME,
    XNET_NAME,
    DRNET_NAME,
    PWNET_NAME,
    RANET_NAME,
    TARNET_NAME,
]


def get_catenet(name: str) -> Any:
    if name not in ALL_MODELS:
        raise ValueError(
            f"Model name should be in catenets.models.ALL_MODELS You passed {name}"
        )
    return MODEL_DICT[name]
