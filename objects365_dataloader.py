import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Literal, Tuple, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
from torchvision.ops import box_convert

from PIL import ImageFile, Image
from PIL.Image import Image as PILImage

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Target(TypedDict):
    image_id:  Tensor
    boxes: Tensor # xyxy
    labels: Tensor
    iscrowd: Tensor
    area: Tensor
    orig_size: Tensor # H, W
    size: Tensor # H, W  
    
def target_set_dtype(target: Target) -> Target:
    target["boxes"] = target["boxes"].float()
    target["labels"] = target["labels"].long()
    target["iscrowd"] = target["iscrowd"].long()
    target["area"] = target["area"].float()
    return target

class Objects365Loader(Dataset):
    def __init__(self,
                 data_root: str | Path,
                 split: str | Path,
                 base_transforms: Optional[Callable[[PILImage, Target | None], Tuple[PILImage, Target | None]]] = None,
                 input_transforms: Optional[Callable[[PILImage, Target | None], Tuple[PILImage, Target | None]]] = None,
                 keep_crowded: bool = True) -> None:

        self.root_dir = Path(data_root) 
        self.split = split
        self.image_dir = self.root_dir / "images" / split
        self.labels_dir = self.root_dir / "labels" / split
        
        self._input_transforms = input_transforms
        self._base_transforms = base_transforms
        self.keep_crowded = keep_crowded
                
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        
        self.dataset = self._load_dataset(self.labels_dir)
        
        self.fail_save = self.__getitem__(0)

    def _load_dataset(self, ann_file):
        patches = os.scandir(self.image_dir)
        dataset = []
        for patch in patches:
            subdir = os.path.join(self.image_dir, patch.name)
            files = os.scandir(subdir)
            
            dataset.extend([f"{patch.name}/{file.name}" for file in files if file.is_file()])
        
        dataset = np.array(dataset).astype(np.string_)
        return dataset

    def _format_annotation(self, annotations, img) -> Target:
        labels = []
        boxes = []
        w, h = img.size

        for line in annotations:
            line = line.split()
            labels.append(int(line[0]))
            boxes.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
            
        boxes = box_convert(torch.as_tensor(boxes, dtype=torch.float32), in_fmt="cxcywh", out_fmt="xyxy") * torch.as_tensor([w, h, w, h])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = Target(**{"labels": labels, "classes": labels, "boxes": boxes, "size": torch.as_tensor([h, w]), "orig_size": torch.as_tensor([h, w])})
        return target
  
    def __len__(self):
        return len(self.dataset)
    
    def _prepare(self, image: PILImage, target_raw):
        w, h = image.size
        size = torch.tensor([int(h), int(w)])
        image_id = torch.tensor(target_raw["image_id"])
        
        ann = target_raw["annotations"]
        if not self.keep_crowded:
            ann = [obj for obj in ann if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in ann])
            
        boxes = [obj["bbox"] for obj in ann] # xywh
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        area = boxes[:, 2] * boxes[:, 3]
        boxes[:, 2:] += boxes[:, :2] # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w) 
        boxes[:, 1::2].clamp_(min=0, max=h)
        
        labels = torch.tensor([obj["category_id"] for obj in ann])
        
        target = Target(image_id = image_id,
                        boxes = boxes,
                        labels=labels,
                        iscrowd=iscrowd,
                        area=area,
                        orig_size=size,
                        size=size
                        )
        
        return image, target
    
    def __getitem__(self, idx: int) -> Tuple[PILImage, Target]:
        ### Load Image and Target data ###
        file = self.dataset[idx]
        file = Path(file.decode("utf-8"))
        
        img_path = self.image_dir / file
        ann_path = self.labels_dir / file
        ann_path = ann_path.with_suffix(".json")
        
        image_id = int(file.stem.split("_")[-1])
                
        with Image.open(img_path) as image:
            image.load()

        with open(ann_path, 'r') as f:
            target_raw = json.load(f)
       
        target_raw = {'image_id': image_id, 'annotations': target_raw}
        image, target = self._prepare(image, target_raw)
        
        if self._base_transforms is not None:
            image, target = self._base_transforms(image, target)
        if self._input_transforms is not None:
            image, target = self._input_transforms(image, target)
            
        target = target_set_dtype(target)

        return image, target
        