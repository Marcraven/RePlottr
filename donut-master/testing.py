from donut import DonutModel
from PIL import Image
import torch

model = DonutModel.from_pretrained("./result/train_plot/test_experiment")

if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)

else:
    model.encoder.to(torch.float32)

model.eval()
image = Image.open("./dataset/train/0005.jpg").convert("RGB")
output = model.inference(image=image, prompt="<s_answer>")
print(output)
