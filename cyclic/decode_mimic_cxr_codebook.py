import torch
import numpy as np
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import tqdm
# from pathlib import Path
# from tkinter import filedialog

from taming.models.vqgan import GumbelVQ, VQModel

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

PATH_CONFIG = "./pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts/2023-05-11T23-37-27-project.yaml" #"<path to the trained model config (.yaml)>"
PATH_CKPT = "./pretrained_model/mimiccxr_vqgan1024_res256_3e_ckpts/last-3e.ckpt" #"<path to the trained model ckpts (.ckpt)>"

torch.set_grad_enabled(False)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = "./tmp/img.png" #Path(filedialog.asksaveasfilename(title="Select save path", defaultextension=".png"))

config = load_config(PATH_CONFIG, display=False)
model = load_vqgan(config, ckpt_path=PATH_CKPT, is_gumbel=False).to(DEVICE)

# indices = eval(input(">> "))
indices = [133, 339, 122, 127, 63, 87, 69, 921, 428, 147, 1002, 63, 203, 258, 193, 122, 163, 428, 807, 621, 468, 127, 961, 322, 147, 63, 15, 921, 932, 637, 955, 439, 930, 295, 808, 824, 773, 1002, 930, 147, 222, 200, 200, 330, 467, 600, 222, 322, 330, 687, 410, 600, 930, 501, 501, 467, 596, 379, 973, 637, 716, 322, 660, 331, 30, 359, 222, 82, 63, 370, 294, 331, 467, 551, 955, 873, 807, 808, 808, 446, 200, 147, 808, 232, 82, 955, 122, 905, 34, 203, 873, 406, 77, 808, 404, 596, 133, 359, 807, 637, 133, 404, 1015, 905, 596, 873, 15, 200, 294, 534, 528, 193, 133, 359, 34, 127, 551, 205, 421, 193, 193, 773, 719, 955, 932, 401, 446, 77, 133, 359, 260, 1015, 333, 807, 205, 401, 330, 885, 232, 406, 404, 108, 905, 446, 133, 69, 333, 961, 860, 258, 978, 860, 790, 428, 665, 200, 329, 69, 528, 119, 22, 687, 551, 333, 250, 151, 202, 596, 22, 428, 322, 742, 406, 222, 528, 105, 22, 764, 243, 203, 930, 163, 687, 446, 22, 909, 764, 339, 555, 147, 665, 529, 107, 932, 529, 468, 467, 905, 764, 205, 292, 447, 845, 108, 22, 807, 406, 955, 151, 428, 144, 808, 330, 330, 930, 439, 930, 292, 637, 785, 842, 508, 322, 961, 151, 439, 345, 401, 30, 108, 955, 439, 719, 999, 790, 30, 331, 905, 294, 955, 151, 122, 829, 687, 808, 108, 551, 66, 860, 133, 785, 439, 905, 82, 573, 410]

assert len(indices) == 256
# indices = torch.tensor(indices).to(DEVICE)

# print("model.quantize.embedding:",model.quantize.embedding)
indice_num = 256 #1024 #model.quantize.embedding.shape
print("indice_num:",indice_num)
all_indices = [i for i in range(indice_num)]
indices = torch.tensor(all_indices).to(DEVICE)
img = model.decode(model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
img = img.squeeze().permute(1,2,0).cpu().numpy()
print("img:",img.max(),img.min())
img = np.clip(img, -1., 1.)
img = (img + 1.)/2.
print("img:",img.shape)
save_path = "./tmp/img1.png"
plt.imsave(save_path, img)


# all_indices = [i+256 for i in range(indice_num)]
# indices = torch.tensor(all_indices).to(DEVICE)
# img = model.decode(model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
# img = img.squeeze().permute(1,2,0).cpu().numpy()
# img = np.clip(img, -1., 1.)
# img = (img + 1.)/2.
# print("img:",img.shape)
# save_path = "./tmp/img2.png"
# plt.imsave(save_path, img)

# all_indices = [i+512 for i in range(indice_num)]
# indices = torch.tensor(all_indices).to(DEVICE)
# img = model.decode(model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
# img = img.squeeze().permute(1,2,0).cpu().numpy()
# img = np.clip(img, -1., 1.)
# img = (img + 1.)/2.
# print("img:",img.shape)
# save_path = "./tmp/img3.png"
# plt.imsave(save_path, img)

# all_indices = [i+768 for i in range(indice_num)]
# indices = torch.tensor(all_indices).to(DEVICE)
# img = model.decode(model.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1)))
# img = img.squeeze().permute(1,2,0).cpu().numpy()
# img = np.clip(img, -1., 1.)
# img = (img + 1.)/2.
# print("img:",img.shape)
# save_path = "./tmp/img4.png"
# plt.imsave(save_path, img)

# img = img.squeeze().permute(1,2,0).cpu().numpy()
# img = np.clip(img, -1., 1.)
# img = (img + 1.)/2.

# plt.imsave(save_path, img)