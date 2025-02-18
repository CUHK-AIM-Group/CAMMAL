from pathlib import Path
from tkinter import filedialog

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from omegaconf import OmegaConf
from PIL import Image
import os
from taming.models.vqgan import GumbelVQ, VQModel
import pandas as pd
from tqdm import tqdm
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Testing
    parser.add_argument('--split_id', type=int,required=True, default=0)
    parser.add_argument('--split_num', type=int,required=True, default=4)
    # Parse the arguments.
    args = parser.parse_args()

    return args
    
def load_image(path, target_image_size=256):
    img = Image.open(path)

    if not img.mode == "RGB":
        img = img.convert("RGB")

    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)
    img = 2.*img - 1.

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


args = parse_args()
split_id = args.split_id
split_num = args.split_num

# root = '~code/img2text2img/llm-cxr/pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts'
root = './pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts/'
# yaml_file = os.path.join(root, '2023-05-11T23-37-27-project.yaml')
# model_path = os.path.join(root, 'last-3e.ckpt')
PATH_CONFIG = os.path.join(root, '2023-05-11T23-37-27-project.yaml') #'2023-05-11T23-37-27-project.yaml') #"<path to the trained model config (.yaml)>"
PATH_CKPT = os.path.join(root, 'last-3e.ckpt') #'last-3e.ckpt') #"<path to the trained model ckpts (.ckpt)>"
RESIZE_SIZE = 256

img_root = '~dataset/x_ray/openI_data/Img'
openI = pd.read_csv('~dataset/x_ray/openI_data/data/openIdf.csv',index_col=0)
openI.head()
id_list = openI.id.tolist()

total = len(id_list)
each_num = total // split_num
start_pos = max(split_id* each_num, 0)
end_pos = min((split_id+1)* each_num, total)
id_list = id_list[start_pos:end_pos]

save_root = "~dataset/x_ray/openI_data/data_new/openI_vqgan1024_codebook_"  + str(split_id) + "_" + str(split_num) + ".pickle"
torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

indice_dict = {}
for idx in tqdm(id_list):
    img_path = os.path.join(img_root, 'CXR'+ str(idx) + '.png')
    # img_path = Path(path)#filedialog.askopenfilename(title="Select load path", defaultextension=".png"))

    config = load_config(PATH_CONFIG, display=False)
    model = load_vqgan(config, ckpt_path=PATH_CKPT, is_gumbel=False).to(DEVICE)

    img = load_image(img_path, RESIZE_SIZE).to(DEVICE)
    _, _, [_, _, indices] = model.encode(img.unsqueeze(0))
    indices = indices.reshape(1, -1).cpu().squeeze().tolist()
    # print(indices)
    # print(len(indices))
    # break
    indice_dict[idx] = indices

# save pickle

with open(save_root, 'wb') as handle:
    pickle.dump(indice_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

