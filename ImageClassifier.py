from pathlib import Path
from typing import List

import yaml

import pandas as pd
import torch
from TestDataProvider import TestDataProvider
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from TrainDataProvider import TrainDataProvider


class ImageClassifier:
    # ==========================================

    def __init__(self, params: dict) -> None:
        self.params = params
        self.parent_path = Path(__file__).parent
        self.training = self.params["is_train"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(params["writername"])
        if self.training:
            self.dataProvider = TrainDataProvider(
                self.params["label"],
                self.parent_path / self.params["path2data"],
                self.params["format_img"],
                self.params["class2label"],
            )
            self.clf_layer = nn.Linear(
                in_features=self.params["feature_size"],
                out_features=len(self.params["class2label"]),
            )
            self._loss = None
            self._optimizer = None
            self._train_dataLoader = None
            self._eval_dataLoader = None
        else:
            self.dataProvider = TestDataProvider(
                self.parent_path / self.params["path2data_test"],
                self.params["format_img"],
                self.params["label2class"],
            )
            self._test_dataLoader = None

    # ==========================================
    @property
    def loss(self) -> nn.functional:
        if self._loss is None:
            if self.params["loss"] == "cross_entropy":
                return nn.CrossEntropyLoss()
            else:
                return None

    # ==========================================
    @property
    def optimizer(self) -> Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.params["lr"]
            )
        return self._optimizer

    # ==========================================
    @property
    def train_dataLoader(self) -> DataLoader:
        if self._train_dataLoader is None:
            self._train_dataLoader = DataLoader(
                self.dataProvider.provide(True), self.params["batch_size"]
            )
        return self._train_dataLoader

    # ==========================================
    @property
    def eval_dataLoader(self) -> DataLoader:
        if self._eval_dataLoader is None:
            self._eval_dataLoader = DataLoader(
                self.dataProvider.provide(False), self.params["batch_size"]
            )
        return self._eval_dataLoader

    # ==========================================
    @property
    def test_dataLoader(self) -> DataLoader:
        if self._test_dataLoader is None:
            self._test_dataLoader = DataLoader(
                self.dataProvider.provide(), self.params["batch_size"]
            )
        return self._test_dataLoader

    # ==========================================
    def build_model(self) -> torch.nn:
        if "vgg16" in self.params["model_name"]:
            model = models.vgg16(pretrained=True)
            model.classifier[-1] = self.clf_layer
            for param in list(model.features.parameters())[
                : self.params["freeze_layer"]
            ]:
                param.require_grad = False
        if "vit" in self.params["model_name"]:
            model = models.vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1")
            model.heads = self.clf_layer
            for param in list(model.parameters())[: self.params["freeze_layer"]]:
                param.require_grad = False
        return model

    # ==========================================
    def load_model(self) -> None:
        self.model = torch.load(self.parent_path / self.params["model_name"])

    # ===========================================
    @staticmethod
    def msg(sen: str) -> None:
        print(sen.center(100, "*"))

    # ==========================================
    def train(self) -> float:
        self.model.train()
        train_loss = 0.0
        for batch, (X, y) in enumerate(self.train_dataLoader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self.model(X)
            loss_t = self.loss(pred, y)
            loss_t.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            train_loss += loss_t.item()

        self.msg(f"train_loss: {train_loss:>4f}")
        return train_loss / len(self.train_dataLoader)

    # ==========================================
    def eval(self) -> float:
        self.model.eval()
        test_loss, metric_value = 0.0, 0.0
        with torch.no_grad():
            for X, y in self.eval_dataLoader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss(pred, y).item()
                metric_value += self.get_metric_value(pred, y)
        test_loss /= len(self.eval_dataLoader)
        metric_value /= len(self.eval_dataLoader.dataset)
        self.msg(
            f"Test Error:  Accuracy: {(100*metric_value):>0.1f}%, Avg loss: {test_loss:>8f} "
        )
        return metric_value

    # ==========================================
    @staticmethod
    def get_metric_value(pred: torch.Tensor, gd: torch.Tensor) -> float:
        return (pred.argmax(1) == gd).type(torch.float).sum().item()

    # ==========================================
    def save_model(self) -> None:
        model_path = self.parent_path / self.params["model_name"]
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, model_path)

    # ==========================================
    def inference(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for X in self.test_dataLoader:
                X = X.to(self.device)
                pred = self.model(X)
        pred_ = list(pred.argmax(1).detach().cpu().type(torch.int).numpy())
        path_list = pd.read_csv(self.dataProvider.path2datacsv)
        for path, ele in zip(path_list["img_path"], pred_):
            self.msg(
                f"inference result of {Path(path).name} is {self.dataProvider.label2class[ele]}"
            )

    # ==========================================
    def log(
        self, log_item_name: List[str], log_item_val: List[float], epoch: int
    ) -> None:
        for name, ite in zip(log_item_name, log_item_val):
            self.writer.add_scalar(name, ite, epoch)

    # ==========================================
    def do_train(self) -> None:
        self.model = self.build_model().to(self.device)
        for epoch in range(self.params["num_epochs"]):
            self.msg(f"{epoch}th_epoch training begins")
            train_loss = self.train()
            accuracy = self.eval()
            self.log(self.params["log_item_name"], [train_loss, accuracy], epoch)
            self.save_model()

    # ==========================================
    def do_inference(self) -> None:
        self.load_model()
        self.msg("Inference !")
        self.inference()

    # ==========================================
    def run(self) -> None:
        if self.training:
            self.do_train()
        else:
            self.do_inference()


# ==========================================
if __name__ == "__main__":
    with open(Path(__file__).parent / "config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    classifier = ImageClassifier(params=cfg)
    classifier.run()
