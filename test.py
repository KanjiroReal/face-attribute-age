from torchinfo import summary

from src.models import AgeRangePredictionVgg13NW

model = AgeRangePredictionVgg13NW(output_shape=5)
summary(model=model, input_size=(32, 3, 224, 224))
