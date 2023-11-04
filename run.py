import os
from pathlib import Path
from time import perf_counter

from PIL import Image
from scipy.spatial import distance
from src.dataset_former import DatasetFormer
from src.owl_runner import OwlRunner

DATASET_LOCATION = Path(os.getenv("DATASET_LOCATION", "data/tasks"))
owl_runner = OwlRunner()
owl_runner.run_all_datasets(DATASET_LOCATION)



