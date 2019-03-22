# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import json
from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file, transforms=None):

        with open(ann_file, "r") as f:
        	self.image_records = json.load(f)

        self.transforms = transforms


    def __getitem__(self, idx):
        rec = self.image_records[idx]
        image_path = rec["path"]

        img = Image.open(image_path).convert("RGB")
        
        boxes = [ [ rec["x1"],
                    rec["y1"],
                    rec["x2"],
                    rec["y2"]  ] ]
        labels = [ rec["label"] ]		# or torch.tensor([ rec["label"] ])

		boxlist = BoxList(boxes, img.size, mode="xyxy")
		boxlist.add_field("labels", labels)

        if self.transforms is not None:
            img, boxlist = self.transforms(img, boxlist)

        return img, boxlist, idx

    def get_img_info(self, idx):
        rec = self.image_records[idx]
        return {
        			"height": rec["height"],
                	"width" : rec["width"]
                }