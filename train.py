import os
from operator import add
from pathlib import Path

import torch as t
from torch import nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchinfo import summary
from torchmetrics import Accuracy, F1Score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms, TrivialAugmentWide

from src import EarlyStopper, Logger, Trainer, AgeRangePredictorVgg16, make_weight_for_balance_classes, \
    get_config


def print_class_counts(data_loader: t.utils.data.DataLoader):
    lst_count = [0] * 5
    for epoch in range(3):
        for i, (inputs, labels) in enumerate(data_loader):
            class_counts = labels.bincount()
            lst_count = list(map(add, lst_count, class_counts.tolist()))
    print(lst_count)


def get_subset_images(dataset, subset):
    return [dataset.imgs[i] for i in subset.indices]


def is_on_kaggle():
    """Check environment, return True when run on Kaggle. else False"""
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return True
    else:
        return False


def main():
    #  Setting agnostic code
    DEVICE = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}\n")
    status = "running on kaggle\n" if is_on_kaggle() else "running on local environment\n"
    print(status)

    # Get config
    config = get_config()
    BATCH_SIZE = 256 if is_on_kaggle() else config.CONTENT.TRAIN.BATCH_SIZE
    IMAGE_SIZE = (224, 224)
    EPOCHS = config.CONTENT.TRAIN.EPOCHS
    NUM_WORKERS = config.CONTENT.DATA.NUM_WORKERS
    TRAIN_SIZE = config.CONTENT.TRAIN.VALIDATION.TRAIN_SIZE
    LR = config.CONTENT.TRAIN.OPTIM.LR
    DATASET_NAME = config.CONTENT.DATA.NAME  # this is the name of the dataset which is the parent of train and test directory
    VERSION = config.PROJECT.VERSION
    print(f"Current version: {VERSION}\n")

    # set up Path
    STORAGE = Path("STORAGE")
    DST_DIR = STORAGE / VERSION
    DST_DIR.mkdir(exist_ok=True, parents=True)

    data_root = Path('/kaggle/input/012-age' if is_on_kaggle() else f'STORAGE/{VERSION}/Data')
    data_path = data_root / DATASET_NAME
    dataset_dir = data_path / "train"
    test_dir = data_path / "test"

    # Set up preprocess
    train_transform = transforms.Compose(
        [transforms.Resize(size=IMAGE_SIZE), TrivialAugmentWide(num_magnitude_bins=10),
         transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])

    # Get dataset and preprocess
    dataset = ImageFolder(dataset_dir, transform=train_transform, target_transform=None)
    class_names = dataset.classes
    test_dataset = ImageFolder(test_dir, transform=train_transform, target_transform=None)
    train_size = int(len(dataset) * TRAIN_SIZE)
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Train size: {len(train_dataset)}\nValidation size: {len(validation_dataset)}")
    print(f"Test size: {len(test_dataset)}\n")

    # get all images
    train_imgs = get_subset_images(dataset, train_dataset)
    validation_imgs = get_subset_images(dataset, validation_dataset)

    # Get weight for imbalanced dataset
    train_weight, _ = make_weight_for_balance_classes(train_imgs, num_classes=len(class_names))
    val_weight, _ = make_weight_for_balance_classes(validation_imgs, num_classes=len(class_names))
    train_weight = t.DoubleTensor(train_weight)
    val_weight = t.DoubleTensor(val_weight)

    # Sampler
    train_sampler = WeightedRandomSampler(weights=train_weight, num_samples=len(train_weight))
    val_sampler = WeightedRandomSampler(weights=val_weight, num_samples=len(val_weight))

    # Data loader with sampler strategy for imbalanced data
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=NUM_WORKERS,
                                  sampler=train_sampler,
                                  drop_last=True)

    val_dataloader = DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=NUM_WORKERS,
                                sampler=val_sampler,
                                drop_last=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=BATCH_SIZE,
                                 num_workers=NUM_WORKERS,
                                 shuffle=False,
                                 drop_last=True)

    data_loader_dict = {"train": train_dataloader,
                        "val": val_dataloader,
                        "test": test_dataloader}

    # Set up model & optimizer & metrics & loss
    model_0 = AgeRangePredictorVgg16(hidden_units=4096,
                                     output_shape=len(class_names)).to(DEVICE)
    model_0_loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    model_0_optimizer = t.optim.Adam(params=model_0.parameters(), lr=LR)
    model_0_lr_scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer=model_0_optimizer, mode='min',
                                                                  patience=config.CONTENT.TRAIN.LR_SCHEDULER.PATIENCE,
                                                                  factor=config.CONTENT.TRAIN.LR_SCHEDULER.FACTOR)
    acc_fn = Accuracy(task="multiclass",
                      num_classes=len(class_names)).to(DEVICE)
    f1_fn = F1Score(task="multiclass",
                    num_classes=len(class_names)).to(DEVICE)

    summary(model_0, input_size=(BATCH_SIZE, 3, 224, 224))

    early_stopper = EarlyStopper(patience_limit=config.CONTENT.TRAIN.ES.PATIENCE,
                                 verbose=True,
                                 model=model_0,
                                 mode=config.CONTENT.TRAIN.ES.MODE,
                                 save_dir=DST_DIR / "Checkpoint")

    log_json = Logger(save_dir=DST_DIR / "Output")
    trainer = Trainer(
        model=model_0,
        dataloader=data_loader_dict,
        loss_func=model_0_loss_fn,
        optimizer=model_0_optimizer,
        lr_scheduler=model_0_lr_scheduler,
        num_epochs=EPOCHS,
        metrics=[acc_fn, f1_fn],
        device=DEVICE,
        checkpoint_dir=DST_DIR / "Checkpoint",
        callbacks=[early_stopper, log_json]
    )
    history = trainer.train()
    print("Trained Successfully")


if __name__ == '__main__':
    main()
