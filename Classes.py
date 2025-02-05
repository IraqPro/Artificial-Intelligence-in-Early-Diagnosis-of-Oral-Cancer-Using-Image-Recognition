import os
import platform
from dataclasses import dataclass

ROOT_PATH = os.path.join(os.getcwd(), "datasets")

@dataclass
class DatasetConfig:
    IMAGE_SIZE: tuple = (384, 384)  # (W, H)
    CHANNELS: int = 3
    NUM_CLASSES: int = 2
    VALID_PCT: float = 0.1

    # Pre-defined MEAN & STD. DEV.of the Imagenet trained model.
    MEAN: tuple = (0.485, 0.456, 0.406)
    STD: tuple = (0.229, 0.224, 0.225)

    # Dataset file and folder paths.
    TRAIN_IMG_DIR: str = os.path.join(ROOT_PATH, "OralCancer", "train")
    TEST_IMG_DIR: str = os.path.join(ROOT_PATH, "OralCancer", "test")
    TRAIN_CSV: str = os.path.join(ROOT_PATH, "OralCancer", "train.csv")
    TEST_CSV: str = os.path.join(ROOT_PATH, "submission.csv")


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 32  # 32, Reduce batch size in case of OOM error.
    NUM_EPOCHS: int = 30
    INIT_LR: float = 1e-4
    NUM_WORKERS: int = 0 if platform.system() == "Windows" else os.cpu_count()
    OPTIMIZER_NAME: str = "Adam"
    WEIGHT_DECAY: float = 1e-4
    USE_SCHEDULER: bool = True  # Use learning rate scheduler?
    SCHEDULER: str = "multi_step_lr"  # Name of the scheduler to use.
    METRIC_THRESH: float = 0.4
    MODEL_NAME: str = "efficientnet_v2_s"  # "efficientnet_v2_s"
    FREEZE_BACKBONE: bool = False

