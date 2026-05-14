"""Argus-AI: Finance-specialised multimodal reasoning model."""

from ml.argus_ai.model import ArgusAI
from ml.argus_ai.backbone import ArgusBackbone
from ml.argus_ai.heads import DirectionHead, SizeHead, TimingHead, ConfidenceHead
from ml.argus_ai.fusion import ModalFusion
from ml.argus_ai.cot_reasoner import ChainOfThoughtReasoner
from ml.argus_ai.rl_tuner import RLTuner
from ml.argus_ai.trainer import ArgusAITrainer

__all__ = [
    "ArgusAI",
    "ArgusBackbone",
    "DirectionHead",
    "SizeHead",
    "TimingHead",
    "ConfidenceHead",
    "ModalFusion",
    "ChainOfThoughtReasoner",
    "RLTuner",
    "ArgusAITrainer",
]

VERSION = "8.37.0"
