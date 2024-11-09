from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class ConstrBeamSearchHyperParams(HyperParams):
    model_name: str
    hf_model: str
    num_beams: int