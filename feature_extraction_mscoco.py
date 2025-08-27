# Tries for dataloader and dataset on feature extraction.

import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from vit import ViT
from prepare_mscoco_dataset import MSCOCODataset


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
PARAMS = {"batch_size": 64, "shuffle": True, "num_workers": 16}

model = ViT().to(device).eval()

mscoco_dt = MSCOCODataset()


class FeatureExtractionDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ids)

    def __getitem__(self, index):
        "Generates one sample of data"
        path = self.ids[index]
        img_id = str(path.stem)
        return path, img_id


def collate_fn(batch):
    paths, ids = zip(*batch)
    return list(paths), list(ids)


class FeatureExtraction:
    def __init__(self, input_folder: Path, output_folder: Path, model: torch.nn.Module, params: dict) -> None:

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model = model
        self.params = params
        self.ids = list(input_folder.glob("*.jpg"))
        self.set = FeatureExtractionDataset(self.ids)
        self.generator = torch.utils.data.DataLoader(self.set, collate_fn=collate_fn, **self.params)
        if not self.output_folder.exists():
            Path(self.output_folder).mkdir(parents=True, exist_ok=True)
            print(f"path: {str(self.output_folder)} is created.")

    def run(self) -> None:
        print(f"Extracting: {str(self.input_folder)}")
        for paths, local_ids in tqdm(self.generator):
            images = [Image.open(p).convert("RGB") for p in paths]
            with torch.no_grad():
                features = self.model(images)
                for feature, id in zip(features, local_ids):
                    torch.save(feature.cpu(), self.output_folder / f"{id}.pt")
        print(f"{str(self.input_folder)} extracted.")


input_folders = [mscoco_dt.train_folder, mscoco_dt.val_folder, mscoco_dt.test_folder]
output_folders = [mscoco_dt.train_features_folder, mscoco_dt.val_features_folder, mscoco_dt.test_features_folder]

for inp, out in zip(input_folders, output_folders):
    x = FeatureExtraction(input_folder=inp, output_folder=out, model=model, params=PARAMS)
    x.run()



