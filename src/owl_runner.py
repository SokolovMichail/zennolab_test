from typing import Dict, List, Any

from PIL import Image
import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection

class OwlRunner:
    processor: Owlv2Processor

    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def predict(self, data: List[Dict[str,Any]]):
        pass