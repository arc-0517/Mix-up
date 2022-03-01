import torch.nn as nn
from torch_trainer.resnet_mixup import resnet18, resnet34, resnet50, resnet101, resnet152

def build_model(model_name:str, n_class:int):

    if model_name == "resnet18":
        model = resnet18(num_classes=n_class)
    elif model_name == "resnet34":
        model = resnet34(num_classes=n_class)
    elif model_name == "resnet50":
        model = resnet50(num_classes=n_class)
    elif model_name == "resnet101":
        model = resnet101(num_classes=n_class)
    elif model_name == "resnet152":
        model = resnet152(num_classes=n_class)
    return model

