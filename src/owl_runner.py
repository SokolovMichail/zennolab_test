from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from PIL import Image
import torch
from scipy.spatial import distance

from transformers import Owlv2Processor, Owlv2ForObjectDetection

from src.dataset_former import DatasetFormer


class OwlRunner:
    processor: Owlv2Processor
    model = Owlv2ForObjectDetection

    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    def run_dataset(self, dataset: List[Dict[str, str | Tuple[float, float]]], dataset_prompt: str):
        dataset_size = len(dataset)
        total_distance = 0.
        true_detections_count = 0.
        for query in dataset:
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
            "true_det_count": true_detections_count,
            "av_euc_dist": total_distance / dataset_size
        }


    def run_all_datasets(self, dataset_folder_path: Path):
        result_dict = {}
        datasets = DatasetFormer.form_datasets(dataset_folder_path)
        for prompt, queries in datasets:
            res_dataset = self.run_dataset(queries, prompt)
            result_dict[prompt] = res_dataset
        with open('report.txt', 'w') as f:
            json.dump(f,result_dict)

    def predict(self, image: Image, prompt: str)->Optional[Tuple[float, Tuple[float,float]]]:
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        target_size_x, target_size_y = int(target_sizes.numpy()[0][0]), int(target_sizes.numpy()[0][1])
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
        return float(t.detach().numpy())

