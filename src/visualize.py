import json
from pathlib import Path
from typing import *

import matplotlib.pyplot as plt
import numpy
from mlxtend.plotting import plot_confusion_matrix
from pandas import DataFrame


class VisualizeLog:
    def __init__(self, dst_dir: Path, src_dir: Path):
        """
    The __init__ function is the constructor for a class. It is called when an object of that class
    is instantiated, and it sets up the attributes of that object. In this case, we are setting up
    the src_dir and dst_dir attributes to be equal to whatever was passed in as arguments.

    :param self: Represent the instance of the class
    :param dst_dir: Path: Create a directory for the destination of the files
    :param src_dir: Path: Specify the directory where the files are located
    :return: Nothing
    """

        self.src_dir = src_dir
        self.dst_dir = dst_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

    def __loadJson(self):
        """
    The __loadJson function loads the json file from the src_dir directory.
        It then returns a dictionary of all data in that json file.

    :param self: Represent the instance of the class
    :return: A dictionary
    """
        json_path = self.src_dir / "result.json"
        with open(json_path, 'r') as f:
            result = json.load(f)
        return result

    def __call__(self) -> bool:
        """
    The __call__ function is used to visualize the curve of the metric in training session from a json format file.
    The function will load the json file and plot each metric with its corresponding validation value.

    :param self: Represent the instance of the class
    :return: True if the function runs successfully
    """

        try:
            results_dict = self.__loadJson()
            for metric_name, metric_value in results_dict.items():
                if "val" in metric_name:
                    continue

                plt.figure(figsize=(20, 7))

                # plotting the train output
                plt.plot(metric_value, label=metric_name, c="green")

                # plotting the val output
                plt.plot(results_dict[f'val_{metric_name}'], label=f"val_{metric_name}", c="orange")

                plt.xlabel("Epochs")
                plt.title(metric_name)
                plt.legend()

                save_path = self.dst_dir / f"{metric_name}.png"
                plt.savefig(save_path)
                print(f"Save fig successfully at path: [{save_path.absolute()}]")
            return True
        except Exception as e:
            print(e)
            return False


def plot_cfs_mat(conf_mat: numpy.ndarray, class_names: List[str], dst_dir: Path, title: str = None) -> None:
    """
The plot_cfs_mat function plots a confusion matrix.

:param conf_mat: numpy.ndarray: Pass the confusion matrix to the function
:param class_names: List[str]: Label the classes in the confusion matrix
:param dst_dir: Path: Specify the directory where the plot will be saved
:param title: str: Set the title of the plot
:return: None
"""

    save_dir = Path(dst_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(conf_mat=conf_mat, class_names=class_names, figsize=(12, 7))
    plt.tight_layout()
    plt.savefig(save_path)
    if title is not None:
        plt.title(title)
    print(f"Save confusion matrix fig successfully at path [{save_path.absolute()}]")
    plt.clf()


def plot_table(df: DataFrame, dst_dir: Path, title: str = None) -> None:
    """
The plot_table function takes a pandas DataFrame and plots it as a table.

:param df: DataFrame: Pass in the dataframe that contains all the evaluation metrics
:param dst_dir: Path: Specify the directory where the plot will be saved
:param title: str: Set the title of the table
:return: None, but saves the table to a file
"""

    table_data = [df.columns.to_list()] + df.values.tolist()
    table = plt.table(cellText=table_data, loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    save_path = dst_dir / "evaluate_metrics_table.png"
    plt.savefig(save_path)
    print(f"Save evaluate matrix_table fig successfully at path [{save_path.absolute()}]")
    plt.clf()
