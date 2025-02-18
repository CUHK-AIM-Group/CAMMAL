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
from mgca.datasets.pretrain_dataset_test import (MultimodalPretrainingDataset,
                                            multimodal_collate_fn)
from mgca.datasets.transforms import DataTransforms
from mgca.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist
from pytorch_lamb import Lamb #, log_lamb_rs
from mgca.models.mgca.scheduler import CosineWarmupScheduler
import pickle
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)

        self.learnable_temp = nn.Parameter(torch.tensor(self.hparams.softmax_temperature))
        # self.result_path = args.result_path
        # print("self.hparams.result_path:",self.hparams.result_path)

        pretrained_model = torch.load(self.hparams.ckpt_path)
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

        
        # learnable 
        # self.img_encoder_q.local_embed.half()
        # self.text_encoder_q.local_embed.half()
        # patch local attention layer
        '''self.patch_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        # sentence local attention layer
        self.word_local_atten_layer = nn.MultiheadAttention(
            self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)'''

        '''self.prototype_layer = nn.Linear(emb_dim, num_prototypes, bias=False)
        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn'''

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

        
        i2t_acc_list, i2t_pre_list, i2t_rec_list, i2t_mrr = self.recall_at_k(
            patch_sim_matrix, labels, top_k=(1, 5, 10))#, batch_size=bs)
        t2i_acc_list, t2i_pre_list, t2i_rec_list, t2i_mrr = self.recall_at_k(
            word_sim_matrix, labels, top_k=(1, 5, 10))#, batch_size=bs)
        
        i2t_acc1, i2t_acc5, i2t_acc10 = i2t_acc_list
        i2t_pre1, i2t_pre5, i2t_pre10 = i2t_pre_list
        i2t_rec1, i2t_rec5, i2t_rec10 = i2t_rec_list 

        t2i_acc1, t2i_acc5, t2i_acc10 = t2i_acc_list
        t2i_pre1, t2i_pre5, t2i_pre10 = t2i_pre_list
        t2i_rec1, t2i_rec5, t2i_rec10 = t2i_rec_list 

        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        acc10 = (i2t_acc10 + t2i_acc10) / 2.

        pre1 = (i2t_pre1 + t2i_pre1) / 2.
        pre5 = (i2t_pre5 + t2i_pre5) / 2.
        pre10 = (i2t_pre10 + t2i_pre10) / 2.

        rec1 = (i2t_rec1 + t2i_rec1) / 2.
        rec5 = (i2t_rec5 + t2i_rec5) / 2.
        rec10 = (i2t_rec10 + t2i_rec10) / 2.

        mrr = (i2t_mrr + t2i_mrr) / 2.

        res_dict = {
            "i2t": { "acc1": i2t_acc1, "acc5": i2t_acc5, "acc10": i2t_acc10, "pre1": i2t_pre1, "pre5": i2t_pre5, "pre10": i2t_pre10, "rec1": i2t_rec1, "rec5": i2t_rec5, "rec10": i2t_rec10, "mrr": i2t_mrr},
            "t2i": { "acc1": t2i_acc1, "acc5": t2i_acc5, "acc10": t2i_acc10, "pre1": t2i_pre1, "pre5": t2i_pre5, "pre10": t2i_pre10, "rec1": t2i_rec1, "rec5": t2i_rec5, "rec10": t2i_rec10, "mrr": t2i_mrr},
            "overall": { "acc1": acc1, "acc5": acc5, "acc10": acc10, "pre1": pre1, "pre5": pre5, "pre5": pre10, "rec1": rec1, "rec5": rec5, "rec5": rec10, "mrr": mrr},
        }


        tokens_sim_matrix_reshape = torch.reshape(tokens_sim_matrix, [bs, patch_token_num, bs, word_token_num])

        return loss_token, loss_token0, loss_token1, acc1, acc5, tokens_sim_matrix_reshape, res_dict #patch_sim_matrix, word_sim_matrix

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

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

    def test_step(self, batch, batch_idx):
        loss_token, loss_token0, loss_token1, acc1, acc5, \
                    tokens_sim_matrix, res_dict = self(batch, batch_idx, "valid")
        # patch_sim_matrix, word_sim_matrix  = self(
        #     batch, batch_idx, "valid")

        # tokens_sim_matrix: bs x patch_token_num x bs x word_token_num
        # information
        # save similarity maps for each image
        bs = tokens_sim_matrix.shape[0]
        device = str(tokens_sim_matrix.device)
        # print("device:",device,type(device))
        pickle_path = os.path.join(self.hparams.result_path, "{}_idx_{}.pickle".format(device, batch_idx))
        # pickle_path = os.path.join(self.result_path, "{}_idx_{}.pickle".format(device, batch_idx))
        item_list = []
        for i, (img_path, sentence, input_ids, special_token_mask , tokens_sim) \
            in enumerate(tqdm(zip(batch['path'],batch['sentences'],batch['caption_ids'],batch['special_tokens_mask'], \
                                                        tokens_sim_matrix), total=bs)):
            cxr_id = img_path.split('/')[-1].split('.')[0] # id
            content = {
                'path': img_path,
                'sentence': sentence,
                'input_ids': input_ids.cpu().detach().numpy(),
                'special_token_mask': special_token_mask.cpu().detach().numpy(),
                'sim': tokens_sim_matrix[i,:,i,:].cpu().detach().numpy(), # patch_token_num x word_token_num
            }
            item_list.append(content)
        # Open a file and use dump()
        with open(pickle_path, 'wb') as file:
            pickle.dump(item_list, file)
        print("Saving the results to ", pickle_path)
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
        
        for direction in res_dict:
            for metric in res_dict[direction]:
                log[direction + "_" + metric] = res_dict[direction][metric]
        
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

    @staticmethod
    def recall_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)): #, batch_size=None):
        device = output.device
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            # print("pred:",pred.shape)#,pred) # 5 x 128
            tgt = target.view(1, -1).expand_as(pred)
            # print("targ:",tgt.shape)#,tgt) # 5 x 128
            sims_normed = F.softmax(output,dim=1)
            # print("sims_normed:",sims_normed)
            # print("batch_size:",batch_size)
            res = []
            precision_list = []
            recall_list = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                # res.append(correct_k.mul_(100.0 / batch_size))
                res.append(correct_k.mul_(1.0 / batch_size))

            labels = []
            for i in range(batch_size):
                tmp = np.zeros((batch_size,))
                tmp[i] = 1
                labels.append(tmp)
            labels = np.array(labels)
            # print("labels:",labels)
            similarities = output.cpu().numpy()

            ranks = [1, 5, 10]
            recall_list, precision_list = [], []
            for k in ranks:
                r_lst, p_lst = [], []
                # for lab, sim in zip(labels, similarities):
                for i in range(similarities.shape[0]):
                    lab = labels[i]
                    sim = similarities[i]
                    sorted_label = []
                    inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                    # print("i:",i,"sim:",sim,"inds:",inds,"lab:",lab)
                    for ind in inds:
                        sorted_label.append(lab[ind])
                    top = np.array(sorted_label[:k]).sum()
                    bottom = np.array(sorted_label).sum()
                    r = top / bottom
                    p = top / k
                    r_lst.append(r)
                    p_lst.append(p)
                r_v = np.mean(np.array(r_lst))
                p_v = np.mean(np.array(p_lst))
                # print("k:",k,"r_v:",r_v,"p_v:",p_v)
                recall_list.append(r_v)
                precision_list.append(p_v)
            # compute rank
            # for lab, sim, idx in zip(labels, similarities, idx_lst):
            ranks = []
            num_txt_per_img = similarities.shape[0]
            for i in range(similarities.shape[0]):
                lab = labels[i]
                sim = similarities[i]
                inds = np.argsort(sim)[::-1]  # descending 4,3,2,1
                rank = num_txt_per_img
                for r, ind in enumerate(inds):
                    if lab[ind] == 1:
                        rank = r
                        break
                ranks.append(rank)
            # compute mrr score
            ranks = np.array(ranks, dtype=float)
            ranks = ranks + 1
            # print('ranks + 1:', ranks)
            mrr_score = np.mean(np.reciprocal(ranks))
            # print('reciprocal_ranks:', np.reciprocal(ranks))
            # print('mrr_score:', mrr_score)
            return res, precision_list, recall_list, mrr_score

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
            warmup_steps= 1000, #self.total_training_steps*0.2, #3000,
            max_steps = self.training_steps,
            lr = self.hparams.learning_rate
        )
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
        # test
        parser.add_argument("--ckpt_path", type=str, default="path for checkpoints")
        parser.add_argument("--result_path", type=str, default="~code/img2text2img/MGCA/data/results")
        
        parser.add_argument("--use_trainset", type=bool, default=False, help="test on training set with test mode")
        
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

    args.deterministic = True
    args.max_epochs = 30 #50

    # save results path
    args.result_path = os.path.join(args.result_path, args.ckpt_path.split('/')[-2])
    os.makedirs(args.result_path, exist_ok=True)

    # seed
    seed_everything(args.seed)

    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn,
                            DataTransforms, args.data_pct,
                            args.batch_size, args.num_workers,
                            use_trainset=args.use_trainset)

    # Add load from checkpoint
    model = MGCA(**args.__dict__)
    # model = MGCA.load_from_checkpoint(args.ckpt_path)#, strict=False)
    # get current time
    # now = datetime.datetime.now(tz.tzlocal())
    # extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    # ckpt_dir = os.path.join(
    #     BASE_DIR, f"../../../data/ckpts/MGCA/{extension}")
    # os.makedirs(ckpt_dir, exist_ok=True)
    # callbacks = [
    #     LearningRateMonitor(logging_interval="step"),
    #     ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
    #                     save_last=True, mode="min", save_top_k=5),
    #     EarlyStopping(monitor="val_loss", min_delta=0.,
    #                   patience=5, verbose=False, mode="min")
    # ]
    # logger_dir = os.path.join(
    #     BASE_DIR, f"../../../data")
    # os.makedirs(logger_dir, exist_ok=True)
    # wandb_logger = WandbLogger(
    #     project="MGCA", save_dir=logger_dir, name=extension)
    # wandb_logger.watch(model, log='all', log_freq=1)
    trainer = Trainer.from_argparse_args(
        args=args)
        # callbacks=callbacks,
        # logger=wandb_logger)

    model.training_steps, model.effective_batch_size = model.num_training_steps(trainer, datamodule)
    print("training_steps:", model.training_steps, "effective_batch_size:", model.effective_batch_size)
    # trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    # best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    # callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()
