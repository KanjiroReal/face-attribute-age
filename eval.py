from pathlib import Path

import pandas as pd
import torch as t
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from src import models, visualize, tools

DEVICE = "cuda" if t.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 0
EVAL_SIZE = 0.0005
# EVAL_SIZE = 0.01109
DATASET_NAME = "012_megaage_up_quantity"  # this is the name of the dataset which is the parent of train and test directory
VERSION = DATASET_NAME[:3]
print(f"Current version: {VERSION}")

STORAGE = Path("STORAGE")
DST_DIR = STORAGE / VERSION
DST_DIR.mkdir(exist_ok=True, parents=True)

data_root = DST_DIR / "Data"
data_path = data_root / DATASET_NAME
dataset_dir = data_path / "train"
test_dir = data_path / "test"

checkpoint_dir = DST_DIR / "Checkpoint"
if __name__ == '__main__':
    # Dataset
    print(f"Getting Dataset...")
    eval_transform = transforms.Compose(
        [transforms.Resize(size=IMAGE_SIZE), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor()])
    eval_data = ImageFolder(root=str(test_dir), transform=eval_transform)
    class_names = eval_data.classes
    print(f"Evaluation size: {len(eval_data)}")

    # Dataloader
    data_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
                             num_workers=NUM_WORKERS)

    # Model & weight
    print("Loading model...")
    model = models.AgeRangePredictorVgg16(hidden_units=1024, output_shape=5)
    model.load_state_dict(t.load(checkpoint_dir / "best_model_epoch_33.pt", map_location=DEVICE))

    # metrics
    acc_fn = Accuracy(task='multiclass', num_classes=5)
    f1_fn = F1Score(task='multiclass', num_classes=5)

    # evaluate
    evaler = tools.Evaler(model=model, data_loader=data_loader, metrics=[acc_fn, f1_fn], device=DEVICE, version=VERSION,
                          class_names=class_names)
    results = evaler()

    curve = visualize.VisualizeLog(src_dir=DST_DIR / "Output", dst_dir=DST_DIR / "Visualization")
    curve()
    visualize.plot_cfs_mat(conf_mat=results["confusion_matrix"].detach().cpu().numpy(), class_names=class_names,
                           dst_dir=DST_DIR / "Visualization", title=f"Confusion matrix VERSION {VERSION}")

    results.pop("confusion_matrix")
    results.pop("version")
    df = pd.DataFrame({"Metrics": [round(metric, 2) for metric in list(results.keys())],
                       "Values": [round(value, 2) for value in list(results.values())]})
    visualize.plot_table(df=df, title=f"Evaluation metric VERSION {VERSION}", dst_dir=DST_DIR / "Visualization")
