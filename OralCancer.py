import os
import shutil
import zipfile
import warnings
import requests
from glob import glob
from itertools import chain
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as TF
from torchvision.utils import make_grid
from torchvision.ops import sigmoid_focal_loss
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchmetrics import MeanMetric
from torchmetrics.classification import MultilabelF1Score
from torchinfo import summary
from Classes import DatasetConfig,TrainingConfig

print(torch.cuda.is_available())
torch.cuda.empty_cache()


torch.set_float32_matmul_precision('high')
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
warnings.filterwarnings(action="ignore", category=UserWarning)
sns.set_palette(sns.color_palette("rocket_r"))

#Labels of Oral Lesions,if suspected
labels = {
    0: 'Cancer',
    1: 'Begnain',
}
rev = {item: key for key, item in labels.items()}

def encode_label(label: list, num_classes=2):
    """This functions converts labels into one-hot encoding"""

    target = torch.zeros(num_classes)
    for l in str(label).split(" "):
        target[int(l)] = 1.0
    return target


def decode_target(
    target: list,
    text_labels: bool = False,
    threshold: float = 0.4,
    cls_labels: dict = None,
):
    """This function converts the labels from
    probablities to outputs or string representations
    """

    result = []
    for i, x in enumerate(target):
        if x >= threshold:
            if text_labels:
                result.append(cls_labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return " ".join(result)


# This function is used for reversing the Normalization step performed
# during image preprocessing.
# Note the mean and std values must match the ones used.

def denormalize(tensors, *, mean, std):
    """Denormalizes image tensors using mean and std provided
    and clip values between 0 and 1"""

    for c in range(DatasetConfig.CHANNELS):
        tensors[:, c, :, :].mul_(std[c]).add_(mean[c])

    return torch.clamp(tensors, min=0.0, max=1.0)


class OralCancerDataset(Dataset):
    """
    Parse raw data to form a Dataset of (X, y).
    """

    def __init__(self, *, df, root_dir, img_size, transforms=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.img_size = img_size
        self.transforms = transforms
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id = row["Image"]
        img_fname = self.root_dir + os.sep + str(img_id) + ".jpeg"

        img = Image.open(img_fname).convert("RGB")
        img = img.resize(self.img_size, resample=3)
        img = self.transforms(img)

        if self.is_test:
            return img, img_id

        return img, encode_label(row["Label"])

class OralCancerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        num_classes=2,
        valid_pct=0.1,
        resize_to=(384, 384),
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle_validation=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.valid_pct = valid_pct
        self.resize_to = resize_to
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_validation = shuffle_validation

        self.train_tfs = TF.Compose(
            [
                TF.RandomAffine(
                    degrees=40,
                    translate=(0.01, 0.12),
                    shear=0.05,
                ),
                TF.RandomHorizontalFlip(),
                TF.RandomVerticalFlip(),
                TF.ToTensor(),
                TF.Normalize(DatasetConfig.MEAN, DatasetConfig.STD, inplace=True),
                TF.RandomErasing(inplace=True),
            ]
        )

        self.valid_tfs = TF.Compose(
            [
                TF.ToTensor(),
                TF.Normalize(DatasetConfig.MEAN, DatasetConfig.STD, inplace=True),
            ]
        )
        self.test_tfs = self.valid_tfs

    def setup(self, stage=None):
        np.random.seed(42)
        data_df = pd.read_csv(DatasetConfig.TRAIN_CSV)
        msk = np.random.rand(len(data_df)) < (1.0 - self.valid_pct)
        train_df = data_df[msk].reset_index()
        valid_df = data_df[~msk].reset_index()

        # train_labels = list(chain.from_iterable([i.strip().split(" ") for i in train_df["Label"].values]))
        # class_weights = compute_class_weight("balanced", classes=list(range(self.num_classes)),
        #                                      y=[int(i) for i in train_labels])
        # self.class_weights = torch.tensor(class_weights)

        img_size = DatasetConfig.IMAGE_SIZE
        self.train_ds = OralCancerDataset(
            df=train_df, img_size=img_size, root_dir=DatasetConfig.TRAIN_IMG_DIR, transforms=self.train_tfs
        )

        self.valid_ds = OralCancerDataset(
            df=valid_df, img_size=img_size, root_dir=DatasetConfig.TRAIN_IMG_DIR, transforms=self.valid_tfs
        )

        test_df = pd.read_csv(DatasetConfig.TEST_CSV)
        self.test_ds = OralCancerDataset(
            df=test_df, img_size=img_size, root_dir=DatasetConfig.TEST_IMG_DIR, transforms=self.test_tfs, is_test=True
        )

        print(f"Number of images :: Training: {len(self.train_ds)}, Validation: {len(self.valid_ds)}, Testing: {len(self.test_ds)}\n")

    def train_dataloader(self):
        # Create a train dataloader.
        train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=True, num_workers=self.num_workers
        )
        return train_loader

    def val_dataloader(self):
        # Create validation dataloader object.
        valid_loader = DataLoader(
            self.valid_ds, batch_size=self.batch_size, pin_memory=self.pin_memory,
            shuffle=self.shuffle_validation, num_workers=self.num_workers
        )
        return valid_loader

    def test_dataloader(self):
        # Create test dataloader object.
        test_loader = DataLoader(
            self.test_ds, batch_size=self.batch_size, pin_memory=self.pin_memory, shuffle=False, num_workers=self.num_workers
        )
        return test_loader


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    """A helper function to load and prepare any classification model
    available in Torchvision for transfer learning or fine-tuning."""

    model = getattr(torchvision.models, model_name)(weights="DEFAULT")

    if freeze_backbone:
        # Set all layer to be non-trainable
        for param in model.parameters():
            param.requires_grad = False

    model_childrens = [name for name, _ in model.named_children()]

    try:
        final_layer_in_features = getattr(model, f"{model_childrens[-1]}")[-1].in_features
    except Exception as e:
        final_layer_in_features = getattr(model, f"{model_childrens[-1]}").in_features

    new_output_layer = nn.Linear(
        in_features=final_layer_in_features,
        out_features=num_classes
    )

    try:
        getattr(model, f"{model_childrens[-1]}")[-1] = new_output_layer
    except:
        setattr(model, model_childrens[-1], new_output_layer)

    return model

model = get_model(
    model_name=TrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    freeze_backbone=False,
)

summary(
    model,
    input_size=(1, DatasetConfig.CHANNELS, *DatasetConfig.IMAGE_SIZE[::-1]),
    depth=2,
    device="cpu",
    col_names=["output_size", "num_params", "trainable"]
)


class OralCancerModel(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            num_classes: int = 2,
            freeze_backbone: bool = False,
            init_lr: float = 0.001,
            optimizer_name: str = "Adam",
            weight_decay: float = 1e-4,
            use_scheduler: bool = False,
            f1_metric_threshold: float = 0.4,
    ):
        super().__init__()

        # Save the arguments as hyperparameters.
        self.save_hyperparameters()

        # Loading model using the function defined above.
        self.model = get_model(
            model_name=self.hparams.model_name,
            num_classes=self.hparams.num_classes,
            freeze_backbone=self.hparams.freeze_backbone,
        )

        # Intialize loss class.
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initializing the required metric objects.
        self.mean_train_loss = MeanMetric()
        self.mean_train_f1 = MultilabelF1Score(num_labels=self.hparams.num_classes,
                                               average="macro", threshold=self.hparams.f1_metric_threshold)
        self.mean_valid_loss = MeanMetric()
        self.mean_valid_f1 = MultilabelF1Score(num_labels=self.hparams.num_classes,
                                               average="macro", threshold=self.hparams.f1_metric_threshold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        data, target = batch
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_train_loss(loss, weight=data.shape[0])
        self.mean_train_f1(logits, target)

        self.log("train/batch_loss", self.mean_train_loss, prog_bar=True)
        self.log("train/batch_f1", self.mean_train_f1, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # Computing and logging the training mean loss & mean f1.
        self.log("train/loss", self.mean_train_loss, prog_bar=True)
        self.log("train/f1", self.mean_train_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def validation_step(self, batch, *args, **kwargs):
        data, target = batch  # Unpacking validation dataloader tuple
        logits = self(data)
        loss = self.loss_fn(logits, target)

        self.mean_valid_loss.update(loss, weight=data.shape[0])
        self.mean_valid_f1.update(logits, target)

    def on_validation_epoch_end(self):
        # Computing and logging the validation mean loss & mean f1.
        self.log("valid/loss", self.mean_valid_loss, prog_bar=True)
        self.log("valid/f1", self.mean_valid_f1, prog_bar=True)
        self.log("step", self.current_epoch)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer_name)(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.init_lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[self.trainer.max_epochs // 2, ],
                gamma=0.1,
            )

            # The lr_scheduler_config is a dictionary that contains the scheduler
            # and its associated configuration.
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "name": "multi_step_lr",
            }
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

        else:
            return optimizer

pl.seed_everything(42, workers=True)

model = OralCancerModel(
    model_name=TrainingConfig.MODEL_NAME,
    num_classes=DatasetConfig.NUM_CLASSES,
    freeze_backbone=TrainingConfig.FREEZE_BACKBONE,
    init_lr=TrainingConfig.INIT_LR,
    optimizer_name=TrainingConfig.OPTIMIZER_NAME,
    weight_decay=TrainingConfig.WEIGHT_DECAY,
    use_scheduler=TrainingConfig.USE_SCHEDULER,
    f1_metric_threshold=TrainingConfig.METRIC_THRESH,
)

data_module = OralCancerDataModule(
    num_classes=DatasetConfig.NUM_CLASSES,
    valid_pct=DatasetConfig.VALID_PCT,
    resize_to=DatasetConfig.IMAGE_SIZE,
    batch_size=TrainingConfig.BATCH_SIZE,
    num_workers=TrainingConfig.NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# Creating ModelCheckpoint callback.
# Checkpoints by default will be saved in Trainer - default_root_dir which is "lightning_logs".
model_checkpoint = ModelCheckpoint(
    monitor="valid/f1",
    mode="max",
    filename="ckpt_{epoch:03d}-vloss_{valid/loss:.4f}_vf1_{valid/f1:.4f}",
    auto_insert_metric_name=False,
)

# Creating a learning rate monitor callback which will be plotted/added in the default logger.
lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

trainer = pl.Trainer(
    accelerator="auto", # Auto select the best hardware accelerator available
    devices="auto", # Auto select available devices for the accelerator (For eg. mutiple GPUs)
    strategy="auto", # Auto select the distributed training strategy.
    max_epochs=TrainingConfig.NUM_EPOCHS, # Maximum number of epoch to train for.
    deterministic=True, # For deteministic and reproducible training.
    enable_model_summary=False, # Disable printing of model summary as we are using torchinfo.
    callbacks=[model_checkpoint, lr_rate_monitor],  # Declaring callbacks to use.
    precision="16", # Using Mixed Precision training.
    logger=True, # Auto generate TensorBoard logs.
)

# Start training
trainer.fit(model, data_module)
CKPT_PATH = model_checkpoint.best_model_path
model = OralCancerModel.load_from_checkpoint(CKPT_PATH)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    enable_checkpointing=False,
    inference_mode=True,
)

# Run evaluation.
data_module.setup()
valid_loader = data_module.val_dataloader()
trainer.validate(model=model, dataloaders=valid_loader)
print(CKPT_PATH)