import json
import os
from pathlib import Path
from typing import *

import torch as t
import torchmetrics
from tqdm.auto import tqdm


class EarlyStopper:
    """
    Early stop if validation loss doesn't improve with a value of `delta_value` after a given number of epochs
    """

    def __init__(self, patience_limit: int, model: t.nn.Module, delta_value: float = 0.0, verbose=False,
                 mode: Literal['val_loss', "loss"] = "val_loss", save_dir: Optional[Path] = None):
        """
        :param patience_limit: Number of epoch to wait
        :param delta_value: A value that validation loss need to improve, if not the training session will stop
        :param verbose:
        :param mode:
        """
        if save_dir is None:
            self.save_dir = Path.cwd()
        else:
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.patience_limit = patience_limit
        self.model = model
        self.delta_value = delta_value
        self.best_score = float("inf")
        self.verbose = verbose
        self.wait = 0
        self.mode = mode
        self.save_path = None

    def __call__(self, result_dict: Dict):
        current_score = result_dict[self.mode][-1]
        cur_epoch = len(result_dict[self.mode])
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.delta_value:
            self.wait += 1
            if self.wait >= self.patience_limit:
                if self.verbose:
                    print(
                        f"Val loss did not improve from {self.best_score} in {self.wait} epochs. Early stop at epoch {cur_epoch}")
                return True
        else:  # val_loss improve
            if self.verbose:
                print(f'Val loss improve from {self.best_score + self.delta_value} → {current_score}!')

            # remove previous checkpoint if save_path is already exist
            if self.save_path is not None:
                os.remove(self.save_path)
                if self.verbose:
                    print(f"Removed previous best checkpoint at path [{self.save_path}]")

            # save best checkpoint
            if self.checkpoint(epoch=cur_epoch):
                if self.verbose:
                    print(f"Successfully saved model at [{self.save_dir}]")
            self.best_score = current_score
            self.wait = 0
            return False

    def checkpoint(self, epoch) -> bool:
        try:
            # update save path
            self.save_path = self.save_dir / f"best_model_epoch_{epoch}.pt"
            t.save(obj=self.model.state_dict(), f=self.save_path)
            return True
        except Exception as e:
            print(e)
            print("\nFailed to save best checkpoint model")
            return False


class Trainer:
    def __init__(self, model: t.nn.Module, dataloader: Dict[str, t.utils.data.DataLoader], loss_func: t.nn.Module,
                 optimizer: t.optim.Optimizer, lr_scheduler: t.optim.lr_scheduler, num_epochs: int,
                 metrics: List[torchmetrics.Metric], device: Any,
                 checkpoint_dir: Union[Path, str] = None, callbacks: List[Callable] = None, re_train: bool = None,
                 re_train_checkpoint: Union[str] = None):
        self.model = model
        self.dataloader = dataloader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.metrics = metrics
        self.device = device
        self.__callbacks = callbacks if callbacks is not None else []
        self.results = {f"{metric.__class__.__name__}": [] for metric in self.metrics}
        self.results.update({f"val_{metric.__class__.__name__}": [] for metric in self.metrics})
        self.loss_dict = {"loss": [], "val_loss": []}
        self.reporter = LossPrettifier(show_percentage=True)
        if checkpoint_dir is None:
            self.checkpoint_dir = Path.cwd()
        else:
            self.checkpoint_dir = checkpoint_dir
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if re_train is not None:
            if re_train_checkpoint is None:
                raise RuntimeError("There are no parameters passed")
            elif not (self.checkpoint_dir / re_train_checkpoint).is_file():
                self.re_train_cp = re_train_checkpoint

    def train(self):
        for epoch in tqdm(range(1, self.num_epochs + 1), desc="TRAIN PROGRESS"):
            self.__train_step(epoch)
            self.__validation_step(epoch)
            self.lr_scheduler.step(self.loss_dict['val_loss'])
            self.__print_loss(epoch)
            if self.__callback():
                break
            self.__save_checkpoint(epoch)
        return {**self.loss_dict, **self.results}

    def __train_step(self, epoch):
        self.model.train()
        total_train_loss = 0
        metrics_values = {metric.__class__.__name__: 0 for metric in self.metrics}
        for batch, (X, y) in enumerate(
                tqdm(self.dataloader["train"],
                     leave=False,
                     desc=f"EPOCH: {epoch} | PHASE: TRAIN")
        ):
            X, y = X.to(self.device), y.to(self.device)
            # Forward
            y_pred = self.model(X)

            # Loss & metrics
            minibatch_loss = self.loss_func(y_pred, y)
            total_train_loss += minibatch_loss
            metrics_values = compute_metrics(preds=y_pred, labels=y, results_dict=metrics_values, metrics=self.metrics)

            # gradient decent
            self.optimizer.zero_grad()
            minibatch_loss.backward()
            self.optimizer.step()

        # Save Final Loss & Metrics
        total_train_loss /= len(self.dataloader["train"])
        metrics_values = {k: v / len(self.dataloader["train"]) for k, v in metrics_values.items()}
        self.loss_dict["loss"].append(total_train_loss.item())
        for k, v in metrics_values.items():
            self.results[k].append(v)

    def __validation_step(self, epoch):
        self.model.eval()
        with t.inference_mode():
            total_val_loss = 0
            val_metrics_values = {f"val_{metric.__class__.__name__}": 0 for metric in self.metrics}
            for batch, (X, y) in enumerate(
                    tqdm(self.dataloader["val"],
                         leave=False,
                         desc=f"EPOCH: {epoch} | PHASE: VALIDATION")
            ):
                X, y = X.to(self.device), y.to(self.device)
                # Forward
                y_pred = self.model(X)

                # Loss & Metrics
                total_val_loss += self.loss_func(y_pred, y)
                val_metrics_values = compute_metrics(preds=y_pred, labels=y, results_dict=val_metrics_values,
                                                     metrics=self.metrics, is_val=True)

            # Save Loss & Metrics
            total_val_loss /= len(self.dataloader["val"])
            val_metrics_values = {k: v / len(self.dataloader["val"]) for k, v in val_metrics_values.items()}
            self.loss_dict["val_loss"].append(total_val_loss.item())
            for k, v in val_metrics_values.items():
                self.results[k].append(v)

    def __save_checkpoint(self, epoch):
        """Checkpoint every epoch and delete previous epoch checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch - 1}.pt"
        if epoch > 1 and checkpoint_path.is_file():
            os.remove(checkpoint_path)
            print(f"Removed epoch {epoch - 1} checkpoint")

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        print(f"Checkpoint Epoch at {self.checkpoint_dir}")
        t.save(obj=self.model.state_dict(), f=checkpoint_path)

    def __callback(self):
        """Execute the list off Callable and return True if it decided to stop the model"""
        stop = False
        if len(self.__callbacks) > 0:
            callback_state = []
            for func in self.__callbacks:
                result = func({**self.loss_dict, **self.results})
                callback_state.append(result)

            if all(callback_state):
                stop = True
        return stop

    def __print_loss(self, epoch):
        self.reporter(epoch=epoch, Loss=self.loss_dict["loss"][-1], Val_loss=self.loss_dict["val_loss"][-1])


class Evaler:
    def __init__(self, model: t.nn.Module, data_loader: t.utils.data.DataLoader, metrics: List[torchmetrics.Metric],
                 device: t.device or str, class_names: List[str], version: str):
        self.model = model
        self.data_loader = data_loader
        self.metrics = metrics
        self.DEVICE = device
        self.class_names = class_names
        self.version = version

    def __call__(self):
        """Return dictionary containing loss, confusion matrix & metric values as keys"""
        self.model.eval()
        results = {"y_pred": [], "y_true": []}
        metrics_values = {metric.__class__.__name__: 0 for metric in self.metrics}
        with t.inference_mode():
            for X, y in tqdm(self.data_loader, desc="EVALUATION"):
                X, y = X.to(self.DEVICE), y.to(self.DEVICE)
                # forward
                y_pred = self.model(X)
                # compute cumulative metric values
                metrics_values = compute_metrics(preds=y_pred, labels=y, metrics=self.metrics,
                                                 results_dict=metrics_values)

                # Track y_pred
                results["y_pred"].append(y_pred.cpu())
                results["y_true"].append(y.cpu())
            metrics_values = {k: v / len(self.data_loader) for k, v in metrics_values.items()}

        y_preds = t.cat(results["y_pred"])
        y_trues = t.cat(results["y_true"])
        conf_matrix = confusion_matrix(y_preds=y_preds, y_true=y_trues, class_names=self.class_names, task="multiclass")
        return {"version": self.version, "confusion_matrix": conf_matrix, **metrics_values}


class Logger:
    def __init__(self, save_dir: Union[Path, str] = None):
        if save_dir is None:
            self.save_dir = Path.cwd()
        else:
            self.save_dir = save_dir
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, result_dic: Dict[str, List[float]]) -> bool:
        try:
            with open(self.save_dir / "result.json", 'w', encoding='utf-8') as f:
                json.dump(result_dic, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(e)
            return False


class LossPrettifier(object):
    STYLE = {'green': '\033[32m', 'red': '\033[91m', 'bold': '\033[1m', }
    STYLE_END = '\033[0m'

    def __init__(self, show_percentage=False):

        self.show_percentage = show_percentage
        self.color_up = 'red'
        self.color_down = 'green'
        self.loss_terms = {}

    def __call__(self, epoch=None, **kwargs):

        if epoch is not None:
            print_string = f'Epoch {epoch: 5d} '
        else:
            print_string = ''

        for key, value in kwargs.items():

            pre_value = self.loss_terms.get(key, value)

            if value > pre_value:
                indicator = '▲'
                show_color = self.STYLE[self.color_up]
            elif value == pre_value:
                indicator = ''
                show_color = ''
            else:
                indicator = '▼'
                show_color = self.STYLE[self.color_down]

            if self.show_percentage:
                show_value = 0 if pre_value == 0 else (value - pre_value) / float(pre_value)
                key_string = f'| {key}: {show_color}{value:3.2f}({show_value:+3.2%}) {indicator}'
            else:
                key_string = f'| {key}: {show_color}{value:.4f} {indicator}'

            # Trim some long outputs
            key_string_part = key_string[:32]
            print_string += key_string_part + f'{self.STYLE_END}\t'

            self.loss_terms[key] = value

        print(print_string)


def walk_through_dir(dir_path):
    """
    walks through a directory and return its contents
    :param dir_path:
    :return:
    """

    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There a {len(dirnames)} directories and {len(filenames)} file in '{dirpath}'")


def compute_metrics(preds: int, labels: int, metrics: List[torchmetrics.Metric], results_dict: Dict[str, float],
                    is_val: bool = False) -> Dict[str, float]:
    """
    Compute cumulative metrics given predictions and labels
    :param is_val: is validation or not, if is_val, it will add prefix "val_" to key of dict
    :param preds: predict
    :param labels: label
    :param metrics: list of metrics
    :param results_dict: Dict[str, float] Dict of metric name and metric value to be computed
    :return:
    """
    prefix = "val_" if is_val else ""
    for metric in metrics:
        metric(preds, labels)
        results_dict[prefix + metric.__class__.__name__] += metric.compute().item()
        metric.reset()
    return results_dict


def confusion_matrix(y_preds: t.Tensor, y_true: t.Tensor, class_names: List[str],
                     task: Literal["binary", "multiclass", "multilabel"]):
    """
    Calculate and return the confusion matrix
    :param y_preds: Predicted value
    :param y_true: True value
    :param class_names: list of class name
    :param task: literal of binary, multiclass and multilabel
    :return: tensor confusion matrix
    """
    conf_mat = torchmetrics.ConfusionMatrix(num_classes=len(class_names), task=task)
    conf_mat_tensor = conf_mat(preds=y_preds, target=y_true)
    return conf_mat_tensor
