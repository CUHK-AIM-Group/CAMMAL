import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
  assert input.size == x1.size == x2.size == x3.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (5*w, h))
  img.paste(input, (0,0))
  img.paste(x0, (1*w,0))
  img.paste(x1, (2*w,0))
  img.paste(x2, (3*w,0))
  img.paste(x3, (4*w,0))
  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font) # coordinates, text, color, font
  return img


def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle: 
      img = map_pixels(img)
    return img

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1,2,0).numpy()
  x = (255*x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  print("z:",z.shape)
  xrec = model.decode(z)
  return xrec

def decode_with_vqgan(z, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  # z, _, [_, _, indices] = model.encode(x)
  # print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  # print("z:",z.shape)
  xrec = model.decode(z)
  return xrec

root = '~/code/image2text2image/CXR2Report2CXR/pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts'
yaml_file = os.path.join(root, '2023-05-11T23-37-27-project.yaml')
model_path = os.path.join(root, 'last-3e.ckpt')

config1024 = load_config(yaml_file, display=False)
model1024 = load_vqgan(config1024, ckpt_path=model_path).cuda()#to(DEVICE)

size = 256
# url='https://heibox.uni-heidelberg.de/f/7bb608381aae4539ba7a/?dl=1'
# x_vqgan = preprocess(download_image(url), target_image_size=size, map_dalle=False)
# x_vqgan = x_vqgan#.to(DEVICE)

# print(f"input is of size: {x_vqgan.shape}")
z = [250, 699, 742, 699, 961, 122, 245, 294, 294, 294, 294, 294, 151, 294, 122, 294, 699, 699, 699, 699, 699, 742, 294, 637, 742, 961, 742, 200, 133, 22, 742, 294, 122, 250, 122, 250, 250, 551, 533, 71, 258, 63, 258, 15, 596, 379, 533, 108, 122, 122, 339, 339, 981, 122, 122, 71, 1012, 421, 764, 401, 634, 534, 716, 873, 258, 200, 981, 961, 955, 1012, 333, 1012, 468, 905, 534, 514, 447, 83, 785, 345, 339, 773, 232, 1012, 885, 222, 359, 467, 534, 119, 193, 22, 200, 596, 34, 66, 932, 66, 105, 447, 127, 222, 322, 127, 129, 468, 22, 1015, 119, 596, 824, 716, 687, 932, 514, 921, 295, 873, 716, 860, 345, 596, 467, 66, 687, 119, 814, 814, 333, 999, 785, 814, 331, 147, 808, 345, 232, 468, 447, 322, 885, 144, 930, 401, 83, 528, 785, 785, 596, 921, 978, 807, 15, 600, 205, 322, 829, 439, 932, 69, 660, 514, 447, 108, 107, 845, 345, 69, 69, 202, 596, 105, 885, 600, 716, 600, 243, 600, 129, 439, 1015, 534, 468, 34, 383, 439, 921, 105, 528, 410, 83, 1012, 333, 873, 421, 807, 1002, 34, 34, 808, 404, 860, 222, 706, 232, 203, 621, 932, 529, 637, 119, 528, 406, 406, 555, 82, 716, 222, 1015, 304, 555, 1012, 439, 999, 764, 818, 1015, 205, 205, 824, 596, 404, 202, 860, 147, 1002, 773, 119, 71, 467, 716, 404, 824, 634, 330, 785, 331, 659, 973, 404, 1002, 410, 829, 147, 873, 339]
# z = [ i for i in range(256)]
z = torch.tensor(z).float().cuda()
print("z:", z.shape)
x0 = decode_with_vqgan(z, model1024)
print("x0:",x0.shape)
img = x0
img = img.squeeze().permute(1,2,0).cpu().numpy()
img = np.clip(img, -1., 1.)
img = (img + 1.)/2.
print("img:",img.shape)
save_path = "./tmp/img1.png"
plt.imsave(save_path, img)
# img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), x3, 
#                             custom_to_pil(x0[0]), custom_to_pil(x1[0]), 
#                             custom_to_pil(x2[0]), titles=titles)
