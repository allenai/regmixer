from enum import Enum


class Metrics(Enum):
    training_loss = "train/CE loss"
    hellaswag = "eval/downstream/hellaswag (length-normalized accuracy)"
