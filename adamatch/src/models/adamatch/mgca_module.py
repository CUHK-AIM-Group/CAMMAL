import datetime
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dateutil import tz
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
import sys 
sys.path.append("../../../")
from mgca.datasets.data_module import DataModule
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.pretrain_dataset_openI import MultimodalPretrainingDataset as MultimodalPretrainingDataset_openI
from mgca.datasets.pretrain_dataset_openI import multimodal_collate_fn as multimodal_collate_fn_openI
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist
from pytorch_lamb import Lamb #, log_lamb_rs
from mgca.models.mgca.scheduler import CosineWarmupScheduler
torch.autograd.set_detect_anomaly(True)
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# from timm.models import create_model
from mgca.models.dpt import dpt_medium


class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 local_temperature: float = 0.1,
                 proto_temperature: float = 0.2,
                 num_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_1: float = 1,
                 lambda_2: float = 0.7,
                 lambda_3: float = 0.5,
                 freeze_prototypes_epochs: int = 1,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.name_img_encoder = img_encoder
        # init encoders
        if img_encoder == 'dpt_medium':
            nb_classes = 13
            drop_rate = 0.0
            drop_path_rate = 0.1
            self.img_encoder_q = dpt_medium(
                pretrained=False,
                num_classes=nb_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                output_dim=self.hparams.emb_dim,
                dpt_return_stage=self.hparams.dpt_return_stage
                # drop_block_rate=None,
            )
            # print("self.img_encoder_q:",self.img_encoder_q)
            # load pretrained model
            pretrained_path = '../../materials/best_model.pth'
            pretrained_model = torch.load(pretrained_path)
            self.img_encoder_q.load_state_dict(pretrained_model['model'], strict=False)
            print("Load pretrained model for {} from {}".format(img_encoder, pretrained_path))
        else:
            self.img_encoder_q = ImageEncoder(
                model_name=img_encoder, output_dim=self.hparams.emb_dim)


        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        self.learnable_temp = nn.Parameter(torch.tensor(self.hparams.softmax_temperature))
        # DEBUG - LOAD PRETRAINED MODEL

        if self.hparams.load_pretrained_model:
            pretrained_model = torch.load(self.hparams.load_pretrained_model)
            state_dict = pretrained_model['state_dict']
            img_encoder_dict = []
            for n, p in pretrained_model['state_dict'].items():
                if 'img_encoder_q' in n:
                    img_encoder_dict.append((n.replace('img_encoder_q.', ''), p))
            text_encoder_dict = []
            for n, p in pretrained_model['state_dict'].items():
                if 'text_encoder_q' in n:
                    text_encoder_dict.append((n.replace('text_encoder_q.', ''), p))

            self.img_encoder_q.load_state_dict(dict(img_encoder_dict))#, strict=False)
            self.text_encoder_q.load_state_dict(dict(text_encoder_dict))#, strict=False)
            self.learnable_temp = nn.Parameter(state_dict['learnable_temp'])

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query image encoder
        img_feat_q, patch_feat_q = self.img_encoder_q(
            batch["imgs"])
        # patch_feat_q = patch_feat_q.half()
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        patch_emb_q = F.normalize(patch_emb_q, p=2, dim=2) #dim=-1)
        # img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        # img_emb_q = F.normalize(img_emb_q, dim=-1)

        # Forward of query text encoder
        report_feat_q, word_feat_q, word_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        # word_feat_q = word_feat_q.half()
        word_emb_q = self.text_encoder_q.local_embed(word_feat_q)
        word_emb_q = F.normalize(word_emb_q, p=2, dim=2)#(word_emb_q, dim=-1)
        # report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        # report_emb_q = F.normalize(report_emb_q, dim=-1)

        bz = word_emb_q.size(0)
        labels = torch.arange(bz).type_as(word_emb_q).long()

        # token similarity among samples
        # patch_emb_q.flatten() # bs x patch_token_num
        # word_emb_q.flatten()  # bs x word_token_num
        bs, patch_token_num, dim1 = patch_emb_q.shape
        _, word_token_num, dim2  = word_emb_q.shape
        patch_emb_q = torch.reshape(patch_emb_q, (bs*patch_token_num, dim1))
        word_emb_q = torch.reshape(word_emb_q, (bs*word_token_num, dim2)).transpose(1,0)
        # print("patch_emb_q:",patch_emb_q.shape)
        # print("word_emb_q:",word_emb_q.shape)
        tokens_sim_matrix = patch_emb_q @ word_emb_q # (bs x patch_token_num) x (bs x word_token_num)
        # print("tokens_sim_matrix:",tokens_sim_matrix.shape)
        # mask for useful word tokens
        word_tokens_mask = 1 - batch["special_tokens_mask"][:,1:] # bs x word_token_num
        # print('batch["special_tokens_mask"] :', batch["special_tokens_mask"].shape)
        # print('batch["special_tokens_mask"][10:15]:',batch["special_tokens_mask"][10:15])
        # print('batch["caption_ids"][10:15]:',batch["caption_ids"][10:15])
        word_tokens_mask = word_tokens_mask.flatten().unsqueeze(0) # 1 x (bs x word_token_num)
        # print("word_tokens_mask:", word_tokens_mask.shape)
        tokens_sim_matrix = tokens_sim_matrix * word_tokens_mask # (bs x patch_token_num) x (bs x word_token_num)
        # get patch_sim_matrix
        patch_sim_matrix = torch.reshape(tokens_sim_matrix, (bs*patch_token_num, bs, word_token_num)) # (bs x patch_token_num) x bs x word_token_num
        patch_sim_matrix = patch_sim_matrix.max(2).values # (bs x patch_token_num) x bs
        patch_sim_matrix = torch.reshape(patch_sim_matrix, [bs, patch_token_num, bs]) # bs x patch_token_num x bs
        patch_sim_matrix = patch_sim_matrix.mean(1) # bs x bs 
        # get word_sim_matrix
        word_sim_matrix = torch.reshape(tokens_sim_matrix, (bs, patch_token_num, bs*word_token_num)) # bs x patch_token_num x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.max(1).values # bs x (bs x word_token_num)
        word_sim_matrix = torch.reshape(word_sim_matrix, [bs, bs, word_token_num]) # bs x (bs x word_token_num)
        word_sim_matrix = word_sim_matrix.sum(2) # bs x bs 
        # batch["cap_lens"] : bs,
        word_sim_matrix = word_sim_matrix / batch["cap_lens"]
        
        patch_sim_matrix /= self.learnable_temp #self.hparams.softmax_temperature
        word_sim_matrix /= self.learnable_temp #self.hparams.softmax_temperature
        # patch_sim_matrix /= self.hparams.softmax_temperature
        # word_sim_matrix /= self.hparams.softmax_temperature
        # print("patch_sim_matrix:",patch_sim_matrix.shape, patch_sim_matrix.dtype, "word_sim_matrix:",word_sim_matrix.shape, word_sim_matrix.dtype)
        # print("patch_sim_matrix:",patch_sim_matrix, "word_sim_matrix:",word_sim_matrix)
        # token_sim = torch.cat(tokens_sim, dim=0) # bs x img_token_num x word_token_num
        
        # patch_token_sim_list = torch.cat(patch_sim_matrix, dim=0)
        # word_token_sim_list = torch.cat(patch_sim_matrix, dim=0)
        # print("patch_token_sim_list:", patch_token_sim_list.shape, "word_token_sim_list:", word_token_sim_list.shape)
        loss_token0 = F.cross_entropy(patch_sim_matrix.to(labels.device), labels)
        loss_token1 = F.cross_entropy(word_sim_matrix.to(labels.device), labels)
        loss_token = loss_token0 + loss_token1 


        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            patch_sim_matrix, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            word_sim_matrix, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.

        return loss_token, loss_token0, loss_token1, acc1, acc5

    def training_step(self, batch, batch_idx):

        # loss_ita, loss_local, loss_proto, acc1, acc5 = self(
        #     batch, batch_idx, "train")
        # loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
        #     loss_local + self.hparams.lambda_3 * loss_proto
        loss_token, loss_token0, loss_token1, acc1, acc5 = self(
            batch, batch_idx, "train")
        # loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
        #     loss_token
        loss = self.hparams.lambda_2 * loss_token

        log = {
            "train_loss": loss,
            # "train_loss_ita": self.hparams.lambda_1 * loss_ita,
            "train_loss_token": self.hparams.lambda_2 * loss_token,
            "train_loss_token_p": self.hparams.lambda_2 * loss_token0,
            "train_loss_token_w": self.hparams.lambda_2 * loss_token1,
            "train_acc1": acc1,
            "train_acc5": acc5
        }
        print("loss_token0:",loss_token0.item(), "loss_token1:", loss_token1.item())
        # log = {
        #     "train_loss": loss,
        #     "train_loss_ita": self.hparams.lambda_1 * loss_ita,
        #     "train_loss_local": self.hparams.lambda_2 * loss_local,
        #     "train_loss_proto": self.hparams.lambda_3 * loss_proto,
        #     "train_acc1": acc1,
        #     "train_acc5": acc5
        # }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)

        return loss

    # freeze prototype layer
    # def on_after_backward(self):
    #     if self.current_epoch < self.hparams.freeze_prototypes_epochs:
    #         for param in self.prototype_layer.parameters():
    #             param.grad = None

    def validation_step(self, batch, batch_idx):
        # loss_ita, loss_local, loss_proto, acc1, acc5 = self(
        #     batch, batch_idx, "valid")

        # loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * \
        #     loss_local + self.hparams.lambda_3 * loss_proto

        # log = {
        #     "val_loss": loss,
        #     "val_loss_ita": self.hparams.lambda_1 * loss_ita,
        #     "val_loss_local": self.hparams.lambda_2 * loss_local,
        #     "val_loss_proto": self.hparams.lambda_3 * loss_proto,
        #     "val_acc1": acc1,
        #     "val_acc5": acc5
        # }
        loss_token, loss_token0, loss_token1, acc1, acc5 = self(
            batch, batch_idx, "valid")

        # loss = self.hparams.lambda_1 * loss_ita + self.hparams.lambda_2 * loss_token
        loss = self.hparams.lambda_2 * loss_token

        log = {
            "val_loss": loss,
            # "val_loss_ita": self.hparams.lambda_1 * loss_ita,
            "val_loss_token": self.hparams.lambda_2 * loss_token,
            "val_loss_token_p": self.hparams.lambda_2 * loss_token0,
            "val_loss_token_w": self.hparams.lambda_2 * loss_token1,
            "val_acc1": acc1,
            "val_acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    # def on_train_epoch_end(self):
    #     ''' Save img_queue and report_queue for visualization '''
    #     if self.local_rank == 0:
    #         img_queue_path = f"{self.trainer.callbacks[-1].dirpath}/img_queue.pth"
    #         torch.save(self.img_queue, img_queue_path)
    #         report_queue_path = f"{self.trainer.callbacks[-1].dirpath}/report_queue.pth"
    #         torch.save(self.report_queue, report_queue_path)

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     self.hparams.learning_rate,
        #     betas=(self.hparams.momentum, 0.999),
        #     weight_decay=self.hparams.weight_decay
        # )

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, )#, torch.nn.Embedding)
        # for mn, m in self.named_modules(): 
        #     for pn, p in m.named_parameters():
        #         fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
        #         print("mn:",mn,"m:","pn:",pn,"fpn:",fpn,"pn.endswith('bias'):",pn.endswith('bias'),"pn.endswith('weight'):",pn.endswith('weight'),"isinstantce:",isinstance(m, blacklist_weight_modules))
        #         if pn.endswith('bias'):
        #             # all biases will not be decayed
        #             no_decay.add(fpn)
        #         elif pn.endswith('weight') and (isinstance(m, blacklist_weight_modules) or "LayerNorm" in pn):
        #             # weights of blacklist modules will NOT be weight decayed
        #             no_decay.add(fpn)
        #         else: 
        #             decay.add(fpn)
        # blacklist = ['pos_embed', 'patch_embed']
        for name, param in self.named_parameters():
            # print("name:",name,"isbias:",name.endswith('bias'),"isweight:",name.endswith('weight'),"is layernorm:",("LayerNorm" in name))#"isinstance:",isinstance(name, blacklist_weight_modules))
            if name.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(name)
            elif name.endswith('weight') and ("LayerNorm" in name or "patch_embed" in name):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(name)
            else: 
                decay.add(name)

        # special case the position embedding parameter in the root GPT module as not decayed
        if self.name_img_encoder == 'dpt_medium':
            no_decay.add('img_encoder_q.pos_embed1')
            decay.remove('img_encoder_q.pos_embed1')

            no_decay.add('img_encoder_q.pos_embed2')
            decay.remove('img_encoder_q.pos_embed2')

            no_decay.add('img_encoder_q.pos_embed3')
            decay.remove('img_encoder_q.pos_embed3')

            no_decay.add('img_encoder_q.pos_embed4')
            decay.remove('img_encoder_q.pos_embed4')
        else:
            no_decay.add('img_encoder_q.model.pos_embed')
            decay.remove('img_encoder_q.model.pos_embed')
        no_decay.add('learnable_temp')
        decay.remove('learnable_temp')

        # print("---decay:",decay)
        # print("---no_decay:",no_decay)
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.hparams.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        # optimizer = torch.optim.AdamW(
        #     optim_groups,
        #     lr=self.hparams.learning_rate,
        #     betas=(self.hparams.momentum, 0.999),
        # )
        # lr_scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=self.training_steps,
        #     cycle_mult=1.0,
        #     max_lr=self.hparams.learning_rate,
        #     min_lr=1e-8,
        #     warmup_steps=int(self.training_steps * 0.4)
        # )

        optimizer = Lamb(optim_groups, \
                        lr=self.hparams.learning_rate, \
                        betas=(.9, .999), \
                        eps=1e-4, \
                        adam=False)

        # total_training_steps = self.training_steps #self.hparams.max_epochs * 
        lr_scheduler = CosineWarmupScheduler(   
            optimizer = optimizer,
            batch_size = self.effective_batch_size,
            warmup_steps= self.training_steps*self.hparams.warmup_step_ratio, #1000, #self.total_training_steps*0.2, #3000,
            max_steps = self.training_steps,
            lr = self.hparams.learning_rate
        )
        print("warmup_step_ratio:",self.hparams.warmup_step_ratio, "warm up steps:", self.training_steps*self.hparams.warmup_step_ratio)
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--softmax_temperature", type=float, default=0.07)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=72) # per gpu
        parser.add_argument("--num_prototypes", type=int, default=500)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--lambda_1", type=float, default=1.)
        parser.add_argument("--lambda_2", type=float, default=1.)
        parser.add_argument("--lambda_3", type=float, default=1.)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--bidirectional", action="store_false")
        parser.add_argument("--data_pct", type=float, default=1.)
        # parser.add_argument("--max_epochs", type=int, default=30)

        parser.add_argument("--dpt_return_stage", type=int, default=4)

        parser.add_argument("--dataset_name", type=str, default="mimic")
        parser.add_argument("--load_pretrained_model", type=str, default=None)

        parser.add_argument("--warmup_step_ratio", type=float, default=0.25)
        
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset) 
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices
        print("dataset_size:",dataset_size,"num_devices:",num_devices,"trainer.accumulate_grad_batches:",trainer.accumulate_grad_batches,"``effective_batch_size``:",effective_batch_size)
        total_training_steps =  (dataset_size // dm.batch_size) * trainer.max_epochs
        return (dataset_size // effective_batch_size) * trainer.max_epochs, effective_batch_size*dm.batch_size


@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def cli_main():
    parser = ArgumentParser()
    # trainer args
    parser = Trainer.add_argparse_args(parser)
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = False #True
    # args.max_epochs = 30 #50

    # seed
    seed_everything(args.seed)

    print("Preparing " + args.dataset_name + " dataset ........")
    if args.dataset_name == 'mimic':
        datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)
    else:
        datamodule = DataModule(MultimodalPretrainingDataset_openI, multimodal_collate_fn_openI,
                                DataTransforms, args.data_pct,
                                args.batch_size, args.num_workers)

    # Add load from checkpoint
    model = MGCA(**args.__dict__)

    # pretrained_model = torch.load('~code/img2text2img/MGCA/pretrained_models/vit_base.ckpt')
    # # model.load_state_dict(pretrained_model['state_dict'], strict=False)
    # state_dict = []
    # # self.model.load_state_dict(dict([(n, p) for n, p in checkpoint['model'].items()]), strict=False)
    # for n, p in pretrained_model['state_dict'].items():
    #     if 'local_embed' in n:
    #         continue
    #     state_dict.append((n, p))
    # model.load_state_dict(dict(state_dict), strict=False)


    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(
        BASE_DIR, f"../../../data/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # save configs
    config_path = os.path.join(ckpt_dir, 'config.txt')
    config_file = open(config_path, 'w')
    # vars_dict = vars(model.hparams)
    # for var in vars_dict:
    #     config_file.write("{} = {}\n".format(var, vars_dict[var]))
    for arg in vars(args):
        config_file.write("{} = {}\n".format(arg, getattr(args, arg)))
    config_file.close()
    print("Config file is saved to {} .....".format(config_path))

    # callbacks = [
    #     LearningRateMonitor(logging_interval="step"),
    #     ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
    #                     save_last=True, mode="min", save_top_k=5),
    #     EarlyStopping(monitor="val_loss", min_delta=0.,
    #                   patience=5, verbose=False, mode="min")
    # ]
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_acc1", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=2),
        ModelCheckpoint(monitor="val_acc5", dirpath=ckpt_dir,
                        save_last=True, mode="max", save_top_k=2),
        # EarlyStopping(monitor="val_loss", min_delta=0.,
        #               patience=5, verbose=False, mode="min")
    ]

    logger_dir = os.path.join(
        BASE_DIR, f"../../../data")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="MGCA", save_dir=logger_dir, name=extension)
    wandb_logger.watch(model, log='all', log_freq=1)
    trainer = Trainer.from_argparse_args(
        args=args,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=1)

    model.training_steps, model.effective_batch_size = model.num_training_steps(trainer, datamodule)
    print("training_steps:", model.training_steps, "effective_batch_size:", model.effective_batch_size)
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)

    best_ckpt_path_acc5 = os.path.join(ckpt_dir, "best_ckpts_acc5.yaml")
    callbacks[2].to_yaml(filepath=best_ckpt_path_acc5)

if __name__ == "__main__":
    cli_main()
