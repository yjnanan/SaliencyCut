import torch
from modeling.layers.deviation_loss import DeviationLoss

def build_criterion(criterion):
    if criterion == "deviation":
        print("Loss : Deviation")
        return DeviationLoss()
    elif criterion == "BCE":
        print("Loss : Binary Cross Entropy")
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "CE":
        print("Loss : CE")
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError