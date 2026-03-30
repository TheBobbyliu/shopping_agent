import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

ckpt = "google/siglip2-so400m-patch14-384"
model = AutoModel.from_pretrained(ckpt, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(ckpt)

image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
inputs = processor(images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs).pooler_output

print(image_embeddings.shape) # torch.Size([1, 1152])
