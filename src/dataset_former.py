from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

class DatasetFormer:
    @staticmethod
    def form_datasets(dataset_folder:Path) -> Dict[str, List[Dict[str, str | Tuple[float, float]]]]:
        """
        Generates dataset in format: {prompt: {"image": path/to/image, "gt": Tuple[float, float] } }
        :param dataset_folder:
        :return:
        """
        dirs = [x for x in dataset_folder.glob("*") if x.is_dir()]
        result = dict()
        for directory in dirs:
            images = directory.glob("*.jpg")
            image_list = list()
            for image in images:
                corresponding_json = Path(directory) / (image.stem + ".json")
                with open(corresponding_json,'r') as f:
                    try:
                        data = json.load(f)
                        if len(data) == 0:
                            continue
                        else:
                            image_list.append(
                                {"image": str(image),
                                 "gt": (data[0]['x'], data[0]['y'])}
                            )
                    except json.decoder.JSONDecodeError:
                        continue
            result[directory.name] = image_list
        return result
