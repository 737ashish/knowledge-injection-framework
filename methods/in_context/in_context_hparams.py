from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class InContextHyperParams(HyperParams):
    model_name: str
    hf_model: str
    split_token: str
    prompt_template: str