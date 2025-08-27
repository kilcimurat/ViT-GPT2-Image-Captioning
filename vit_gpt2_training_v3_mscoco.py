"""Fine-tune BLIP-2 on MSCOCO with multi-GPU support and beam search decoding."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from torch.optim import AdamW
from tqdm import tqdm

from prepare_mscoco_dataset import MSCOCODataset
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


# Reproducibility
torch.manual_seed(3)
torch.cuda.manual_seed_all(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dt = MSCOCODataset()
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
model = model.to(DEVICE)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=1e-5)


annotation_file = dt.val_captions
annotation_name = Path(annotation_file).stem
coco = COCO(str(annotation_file))

train_data, val_data, _ = dt.load_data()
train_paths, train_captions, _ = zip(*train_data)
val_paths, val_ids = zip(*val_data)


class TrainDataset(Dataset):
    def __init__(self, paths, captions, root):
        self.paths = paths
        self.captions = captions
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.root / f"{self.paths[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        caption = self.captions[idx]
        return image, caption


class ValDataset(Dataset):
    def __init__(self, paths, ids, root):
        self.paths = paths
        self.ids = ids
        self.root = root

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.root / f"{self.paths[idx]}.jpg"
        image = Image.open(img_path).convert("RGB")
        return image, self.ids[idx]


def train_collate(batch):
    images, captions = zip(*batch)
    inputs = processor(images=list(images), text=list(captions), padding=True, return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def val_collate(batch):
    images, ids = zip(*batch)
    inputs = processor(images=list(images), return_tensors="pt")
    return inputs, ids


train_loader = DataLoader(
    TrainDataset(train_paths, train_captions, dt.train_folder),
    batch_size=16,
    shuffle=True,
    num_workers=8,
    collate_fn=train_collate,
)

val_loader = DataLoader(
    ValDataset(val_paths, val_ids, dt.val_folder),
    batch_size=32,
    shuffle=False,
    num_workers=8,
    collate_fn=val_collate,
)


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    for batch in tqdm(train_loader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()
    return losses / len(train_loader)


def generate_captions(model, pixel_values):
    model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
    generated_ids = model_to_use.generate(pixel_values=pixel_values, num_beams=5, max_new_tokens=30)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)


def test_epoch(model, best_score, epoch):
    model.eval()
    data = []
    with torch.no_grad():
        for batch, ids in tqdm(val_loader):
            pixel_values = batch["pixel_values"].to(DEVICE)
            captions = generate_captions(model, pixel_values)
            for caption, img_id in zip(captions, ids):
                data.append({"image_id": img_id, "caption": caption})

    json_file = f"results/{annotation_name}_result.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    coco_result = coco.loadRes(json_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params["image_id"] = coco_result.getImgIds()
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")
        if metric == "CIDEr" and score > best_score:
            best_score = score
            with open(f"results/best_{annotation_name}_result.json", "w") as f:
                json.dump(data, f)

    return best_score


NUM_EPOCHS = 40
BEST_CIDER_SCORE = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, optimizer)
    BEST_CIDER_SCORE = test_epoch(model, BEST_CIDER_SCORE, epoch)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}")
    with open("best_cider_score.txt", "w") as f:
        f.write(f"Best CIDEr Score: {BEST_CIDER_SCORE}")

