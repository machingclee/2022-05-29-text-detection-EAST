from models import EAST
from glob import glob
from device import device
from PIL import Image
from detect import detect, plot_boxes
import torch

model_path = './pths/model_epoch_5.pth'
model = EAST().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
img_paths = glob("test/in/*.jpg")

for img_path in img_paths:
    result_img_path = img_path.replace("/in", "/out").replace(".jpg", "_result.jpg")
    img = Image.open(img_path)
    boxes = detect(img, model, device)
    plot_img = plot_boxes(img, boxes)
    plot_img.save(result_img_path)