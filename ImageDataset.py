import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    # ==============================================================================
    def __init__(
        self, input_df: pd.DataFrame, resize: int = 224, train: bool = True
    ) -> None:
        self.df_img = input_df
        self._images_labels = []
        self.resize = resize
        self.train = train
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # ==============================================================================
    @property
    def images_labels(self) -> None:
        if len(self._images_labels) == 0:
            if self.train:
                for img_path, label in zip(
                    self.df_img["img_path"], self.df_img["label"]
                ):
                    self._images_labels.append((Image.open(str(img_path)), label))
            else:
                for img_path in self.df_img["img_path"]:
                    self._images_labels.append(Image.open(str(img_path)))

        return self._images_labels

    # ==============================================================================
    def __len__(self) -> int:
        return len(self.images_labels)

    # ==============================================================================
    def __getitem__(self, item):
        if self.train:
            return (
                self.transform(self.images_labels[item][0]),
                self.images_labels[item][1],
            )
        else:
            return self.transform(self.images_labels[item])
