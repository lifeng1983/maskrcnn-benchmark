import cv2
import torch
import requests

import numpy as np

from torchvision import transforms as T
from io import BytesIO
from PIL import Image
from time import perf_counter
from maskrcnn_benchmark.structures.image_list import to_image_list

CATEGORIES = [
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

#config
TO_BGR255 = True
PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
PIXEL_STD = [1.0, 1.0, 1.0]
SIZE_DIVISIBILITY = 32

#config
min_image_size = 800
confidence_threshold = 0.2


def load(url):
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def build_transform():
    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if TO_BGR255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    normalize_transform = T.Normalize(
                                mean=PIXEL_MEAN,
                                std=PIXEL_STD
                            )

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size),
            T.ToTensor(),
            to_bgr_transform,
            normalize_transform,
        ]
    )
    return transform

transforms = build_transform()
device = torch.device("cpu")
cpu_device = torch.device("cpu")


def compute_prediction(model, original_image):
    # apply pre-processing to image
    image = transforms(original_image)
    image_list = to_image_list(image, SIZE_DIVISIBILITY)
    image_list = image_list.to(device)
    # compute predictions
    with torch.no_grad():
        predictions = model(image_list)
    predictions = [o.to(cpu_device) for o in predictions]

    # always single image is passed at a time
    prediction = predictions[0]

    # reshape prediction (a BoxList) into the original image size
    height, width = original_image.shape[:-1]
    prediction = prediction.resize((width, height))

    return prediction

def select_top_predictions(predictions, threshold):
    #threshold = confidence_threshold
        
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


image = load(
        "https://timedotcom.files.wordpress.com/2018/09/neon-brights-street-style.jpg"
    )

model = torch.load("/home/fenli/my_model/fashion_category_attr/saved_model.pth")
model.eval()
model.to(device)

start_time = perf_counter()

predictions = compute_prediction(model, image)

selected_predictions = select_top_predictions(predictions, confidence_threshold)

boxes = selected_predictions.bbox
scores = selected_predictions.get_field("scores").tolist()
labels = selected_predictions.get_field("labels").tolist()
labels = [CATEGORIES[i] for i in labels]

for box, score, label in zip(boxes, scores, labels):
    x1,y1,x2,y2=box[:4]
    print("{:<20} {:>6.2f} {:>6.0f} * {:.0f}".format(label, score, x2-x1, y2-y1))

elapsed_time = perf_counter() - start_time
print("Done in {:.0f} ms".format(elapsed_time * 1000))