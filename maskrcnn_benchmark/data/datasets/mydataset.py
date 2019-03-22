import torch
import json

from PIL import Image
from maskrcnn_benchmark.structures.bounding_box import BoxList

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, ann_file, transforms=None):
        self.data_dir = data_dir
        with open("{}/{}".format(data_dir, ann_file), "r") as f:
            self.image_records = json.load(f)
        self.transforms = transforms
        
    
    def __len__(self):
        return len(self.image_records)
    
    def __getitem__(self, idx):
        rec = self.image_records[idx]

        file_path = "{}/{}".format(self.data_dir, rec["path"])
        #print(file_path)
        img = Image.open( file_path ).convert("RGB")
        
        boxes = [ [ rec["x1"],
                    rec["y1"],
                    rec["x2"],
                    rec["y2"]  ] ]
        labels = torch.tensor([ rec["label"] ])

        boxlist = BoxList(boxes, img.size, mode="xyxy")
        boxlist.add_field("labels", labels)

        if self.transforms is not None:
            img, boxlist = self.transforms(img, boxlist)

        return img, boxlist, idx

    def get_img_info(self, idx):
        rec = self.image_records[idx]
        return { "height": rec["height"],
                 "width" : rec["width"] }