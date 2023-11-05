from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from PIL import Image
import torch
from scipy.spatial import distance
from tqdm import tqdm

import os

from transformers import OwlViTProcessor, OwlViTForObjectDetection

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:30"

#DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from src.dataset_former import DatasetFormer


class OwlRunner:

    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", load_in_8bit=True)
        #self.model.to(DEVICE)

    def run_dataset(self, dataset: List[Dict[str, str | Tuple[float, float]]], dataset_prompt: str):
        dataset_size = len(dataset)
        total_distance = 0.
        true_detections_count = 0.
        for query in tqdm(dataset):
            image = Image.open(query['image'])
            prediction = self.predict(image, dataset_prompt)
            if prediction is not None:
                center = prediction[1]
                euc_dist = distance.euclidean(center, query['gt'])
            else:
                euc_dist = 1.
            if euc_dist < 0.1:
                true_detections_count += 1
            total_distance += euc_dist
        return {
            "dataset_precision": true_detections_count / dataset_size,
            "dataset_av_euc_dist": total_distance / dataset_size,
            "distance" : total_distance,
            "true_detections_count": true_detections_count,
            "dataset_size": dataset_size
        }


    def run_all_datasets(self, dataset_folder_path: Path):
        result_dict = {}
        datasets = DatasetFormer.form_datasets(dataset_folder_path)
        for prompt, dataset in datasets.items():
            res_dataset = self.run_dataset(dataset, prompt)
            result_dict[prompt] = res_dataset
        with open('report.txt', 'w') as f:
            json.dump(result_dict,f)

    def predict(self, image: Image, prompt: str)->Optional[Tuple[float, Tuple[float,float]]]:
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        inputs.to(DEVICE)
        torch.cuda.empty_cache()
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]],)
        target_size_x, target_size_y = int(target_sizes.numpy()[0][0]), int(target_sizes.numpy()[0][1])
        target_sizes = target_sizes.to(DEVICE)
        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
        i = 0  # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        zipped = [(x, y, z) for x, y, z in zip(boxes, scores, labels)]
        zipped.sort(key=lambda x: x[1], reverse=True)
        if len(zipped) > 0:
            result = zipped[0]
            box = result[0]
            centroid = (OwlRunner.float_tensor_to_numpy((box[0] + box[2]) / 2 / target_size_x),
                        OwlRunner.float_tensor_to_numpy((box[1] + box[3]) / 2 / target_size_y))
            return OwlRunner.float_tensor_to_numpy(result[1]), centroid

    @staticmethod
    def float_tensor_to_numpy(t: torch.Tensor) -> float:
        return float(t.cpu().detach().numpy())

