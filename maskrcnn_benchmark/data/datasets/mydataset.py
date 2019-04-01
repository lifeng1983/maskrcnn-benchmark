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
    
    def get_groundtruth(self, idx):
        rec = self.image_records[idx]
        
        boxes = [ [ rec["x1"],
                    rec["y1"],
                    rec["x2"],
                    rec["y2"]  ] ]
    
        target = BoxList(boxes, (rec["width"], rec["height"]), mode="xyxy")
        target.add_field("labels", torch.tensor([ rec["label"] ]))
        target.add_field("difficult", torch.tensor([ False ]))

        return target
    
    def map_class_id_to_class_name(self, class_id):
        return MyDataset.CLASSES[class_id]
    
    
    CLASSES = [
        "__background",
        "Anorak",
        "Blazer",
        "Blouse",
        "Bomber",
        "Button-Down",
        "Cardigan",
        "Flannel",
        "Halter",
        "Henley",
        "Hoodie",
        "Jacket",
        "Jersey",
        "Parka",
        "Peacoat",
        "Poncho",
        "Sweater",
        "Tank",
        "Tee",
        "Top",
        "Turtleneck",
        "Capris",
        "Chinos",
        "Culottes",
        "Cutoffs",
        "Gauchos",
        "Jeans",
        "Jeggings",
        "Jodhpurs",
        "Joggers",
        "Leggings",
        "Sarong",
        "Shorts",
        "Skirt",
        "Sweatpants",
        "Sweatshorts",
        "Trunks",
        "Caftan",
        "Cape",
        "Coat",
        "Coverup",
        "Dress",
        "Jumpsuit",
        "Kaftan",
        "Kimono",
        "Nightdress",
        "Onesie",
        "Robe",
        "Romper",
        "Shirtdress",
        "Sundress"
    ]