import os
from pathlib import Path
from src.owl_runner import OwlRunner

DATASET_LOCATION = Path(os.getenv("DATASET_LOCATION", "data/tasks"))
owl_runner = OwlRunner()
owl_runner.run_all_datasets(DATASET_LOCATION)



