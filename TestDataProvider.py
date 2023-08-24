from pathlib import Path

import pandas as pd
from ImageDataset import ImageDataset
from PIL import Image
from torch.utils.data import Dataset


class TestDataProvider:
    # ==========================================
    def __init__(self, path2data: Path, format_img: str, label2class: dict) -> None:
        self.path2data = path2data
        self._data = pd.DataFrame()
        self.format_img = format_img
        self.label2class = label2class
        self.path2datacsv=self.path2data/"test.csv"

    # ==========================================
    def convert2png(self) -> None:
        
            img_path_list = []

            for ext in ("*.jpg", "*.jpeg"):
                img_path_list.extend(self.path2data.glob(ext))
            if len(img_path_list) != 0:
                [
                    (
                        Image.open(path_im).save(
                            self.path2data
                            / (Path(path_im).stem + f".{self.format_img}")
                        ),
                        Path(path_im).unlink(),
                    )
                    for path_im in img_path_list
                ]

    # ==========================================
    def set_data(self) -> None:
        self.convert2png()
        
        path = self.path2datacsv
        if not path.is_file():
            frame = pd.DataFrame(
                data=list(path.parent.rglob(f"*.{self.format_img.lower()}")),
                columns=["img_path"],
            )
            frame.to_csv(path)
        else:
            print(f"{path} exists already")

    # ===========================================
    def load_data(self) -> pd.DataFrame:
        self.set_data()
        return pd.concat(list(map(pd.read_csv, self.path2data.rglob("*.csv"))), axis=0)

    # ==========================================
    @property
    def data(self) -> pd.DataFrame:
        if len(self._data) == 0:
            self._data = self.load_data()
        return self._data

    # ==========================================
    def provide(self) -> Dataset:
        return ImageDataset(self.data, train=False)
