import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: typehints
class ClassificationModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_p, num_classes):
        super().__init__()
        # Hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p
        self.num_classes = num_classes

        # Define layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        z = self.fc1(batch["embedding"])
        z = self.batch_norm(z)
        z = self.relu(z)
        z = self.dropout(z)
        z = self.fc2(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_probabilities(self, batch):
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        Path(dp).mkdir(parents=True, exist_ok=True)
        with open(Path(dp, "args.json"), "w") as fp:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "dropout_p": self.dropout_p,
                    "num_classes": self.num_classes,
                },
                fp,
                indent=4,
            )
        torch.save(self.state_dict(), Path(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp, device="cpu"):
        with open(args_fp, "r") as fp:
            model = cls(**json.load(fp))
        model.load_state_dict(torch.load(state_dict_fp, map_location=device))
        return model
