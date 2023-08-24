from pathlib import Path

import pandas as pd
from ImageDataset import ImageDataset
from PIL import Image
from torch.utils.data import Dataset


class TrainDataProvider:
    # ==========================================
    def __init__(
        self,
        label: str,
        path2data: Path,
        format_img: str,
        class2label: dict,
        split_ratio: float = 0.75,
    ) -> None:
        self.path2data = path2data
        self.split_ratio = split_ratio
        self._data = pd.DataFrame()
        self._data_train_length = 0
        self.label = label
        self.format_img = format_img
        self.class2label = class2label

    # ==========================================
    @property
    def data_train_length(self) -> int:
        if self._data_train_length == 0:
            self._data_train_length = int(len(self.data) * self.split_ratio)
        return self._data_train_length

    # ==========================================
    def convert2png(self) -> None:
        for key in self.class2label:
            img_path_list = []

            for ext in ("*.jpg", "*.jpeg"):
                img_path_list.extend(self.get_dataset_path(key).glob(ext))
            if len(img_path_list) != 0:
                [
                    (
                        Image.open(path_im).save(
                            self.get_dataset_path(key)
                            / (Path(path_im).stem + f".{self.format_img}")
                        ),
                        Path(path_im).unlink(),
                    )
                    for path_im in img_path_list
                ]

    # ==========================================
    def get_dataset_csv_path(self, key: str) -> Path:
        return self.get_dataset_path(key) / f"{key}.csv"

    # ==========================================
    def get_dataset_path(self, key: str) -> Path:
        return (self.path2data / key).resolve()

    # ==========================================
    def set_data(self) -> None:
        self.convert2png()
        for key in self.class2label:
            path = self.get_dataset_csv_path(key)
            if not path.is_file():
                frame = pd.DataFrame(
                    data=list(path.parent.rglob(f"*.{self.format_img.lower()}")),
                    columns=["img_path"],
                )
                frame[self.label] = self.class2label[key]
                frame.to_csv(path)
            else:
                print(f"{path} exists already")

    # ===========================================
    def load_data(self) -> pd.DataFrame:
        self.set_data()
        return pd.concat(
            list(map(pd.read_csv, self.path2data.rglob("*.csv"))), axis=0
        ).sample(frac=1)

    # ==========================================
    @property
    def data(self) -> pd.DataFrame:
        if len(self._data) == 0:
            self._data = self.load_data()
        return self._data

    # ==========================================
    def provide(self, train: bool) -> Dataset:
        if train:
            return ImageDataset(self.data.iloc[0 : self.data_train_length])
        else:
            return ImageDataset(self.data.iloc[self.data_train_length :])
