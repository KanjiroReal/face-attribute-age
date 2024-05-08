from torch import nn
from torchvision.models import vgg16, vgg13


class AgeRangePredictorVgg16(nn.Module):
    def __init__(self, hidden_units: int = 1024, output_shape: int = 5):
        """
        The __init__ function is called when the object is instantiated. It sets up the layers of our network,
        backbone and classifier, as well as a few bookkeeping items like hidden_units and output_shape.

    :param self: Represent the instance of the class
    :param hidden_units: int: Set the number of neurons in the hidden layers
    :param output_shape: int: Specify the number of classes in the dataset
    :return: Nothing
    """
        super().__init__()
        self.backbone = vgg16(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        backbone_output_shape = self.backbone[0][-3].out_channels
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=backbone_output_shape * 7 * 7, out_features=hidden_units),
                                        nn.ReLU(), nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                        nn.ReLU(), nn.Linear(in_features=hidden_units, out_features=output_shape))

    def forward(self, x):
        return self.classifier(self.backbone(x))


class AgeRangePredictionVgg13NW(nn.Module):
    def __init__(self, output_shape: int):
        super().__init__()
        self.backbone = vgg13()
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape),
            nn.Softmax()
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
