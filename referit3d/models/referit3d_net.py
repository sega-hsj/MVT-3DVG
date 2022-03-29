import torch
import argparse
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from . import DGCNN
from .utils import get_siamese_features, my_get_siamese_features
from ..in_out.vocabulary import Vocabulary
import math
try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None

from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from transformers import BertTokenizer, BertModel, BertConfig
from referit3d.models import MLP
import time


class ReferIt3DNet_transformer(nn.Module):

    def __init__(self,
                 args,
                 n_obj_classes,
                 class_name_tokens,
                 ignore_index):

        super().__init__()

        self.bert_pretrain_path = args.bert_pretrain_path

        self.view_number = args.view_number
        self.rotate_number = args.rotate_number

        self.label_lang_sup = args.label_lang_sup
        self.aggregate_type = args.aggregate_type

        self.encoder_layer_num = args.encoder_layer_num
        self.decoder_layer_num = args.decoder_layer_num
        self.decoder_nhead_num = args.decoder_nhead_num

        self.object_dim = args.object_latent_dim
        self.inner_dim = args.inner_dim
        
        self.dropout_rate = args.dropout_rate
        self.lang_cls_alpha = args.lang_cls_alpha
        self.obj_cls_alpha = args.obj_cls_alpha

        self.object_encoder = PointNetPP(sa_n_points=[32, 16, None],
                                        sa_n_samples=[[32], [32], [None]],
                                        sa_radii=[[0.2], [0.4], [None]],
                                        sa_mlps=[[[3, 64, 64, 128]],
                                                [[128, 128, 128, 256]],
                                                [[256, 256, self.object_dim, self.object_dim]]])

        self.language_encoder = BertModel.from_pretrained(self.bert_pretrain_path)
        self.language_encoder.encoder.layer = BertModel(BertConfig()).encoder.layer[:self.encoder_layer_num]

        self.refer_encoder = nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=self.inner_dim, 
            nhead=self.decoder_nhead_num, dim_feedforward=2048, activation="gelu"), num_layers=self.decoder_layer_num)

        # Classifier heads
        self.language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                        nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                        nn.Linear(self.inner_dim, n_obj_classes))

        self.object_language_clf = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), 
                                                nn.ReLU(), nn.Dropout(self.dropout_rate), 
                                                nn.Linear(self.inner_dim, 1))

        if not self.label_lang_sup:
            self.obj_clf = MLP(self.inner_dim, [self.object_dim, self.object_dim, n_obj_classes], dropout_rate=self.dropout_rate)

        self.obj_feature_mapping = nn.Sequential(
            nn.Linear(self.object_dim, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.box_feature_mapping = nn.Sequential(
            nn.Linear(4, self.inner_dim),
            nn.LayerNorm(self.inner_dim),
        )

        self.class_name_tokens = class_name_tokens

        self.logit_loss = nn.CrossEntropyLoss()
        self.lang_logits_loss = nn.CrossEntropyLoss()
        self.class_logits_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @torch.no_grad()
    def aug_input(self, input_points, box_infos):
        input_points = input_points.float().to(self.device)
        box_infos = box_infos.float().to(self.device)
        xyz = input_points[:, :, :, :3]
        bxyz = box_infos[:,:,:3] # B,N,3
        B,N,P = xyz.shape[:3]
        rotate_theta_arr = torch.Tensor([i*2.0*np.pi/self.rotate_number for i in range(self.rotate_number)]).to(self.device)
        view_theta_arr = torch.Tensor([i*2.0*np.pi/self.view_number for i in range(self.view_number)]).to(self.device)
        
        # rotation
        if self.training:
            # theta = torch.rand(1) * 2 * np.pi  # random direction rotate aug
            theta = rotate_theta_arr[torch.randint(0,self.rotate_number,(B,))]  # 4 direction rotate aug
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotate_matrix = torch.Tensor([[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,1.0]]).to(self.device)[None].repeat(B,1,1)
            rotate_matrix[:, 0, 0] = cos_theta
            rotate_matrix[:, 0, 1] = -sin_theta
            rotate_matrix[:, 1, 0] = sin_theta
            rotate_matrix[:, 1, 1] = cos_theta

            input_points[:, :, :, :3] = torch.matmul(xyz.reshape(B,N*P,3), rotate_matrix).reshape(B,N,P,3)
            bxyz = torch.matmul(bxyz.reshape(B,N,3), rotate_matrix).reshape(B,N,3)
        
        # multi-view
        bsize = box_infos[:,:,-1:]
        boxs=[]
        for theta in view_theta_arr:
            rotate_matrix = torch.Tensor([[math.cos(theta), -math.sin(theta), 0.0],
                                        [math.sin(theta), math.cos(theta),  0.0],
                                        [0.0,           0.0,            1.0]]).to(self.device)
            rxyz = torch.matmul(bxyz.reshape(B*N, 3),rotate_matrix).reshape(B,N,3)
            boxs.append(torch.cat([rxyz,bsize],dim=-1))
        boxs=torch.stack(boxs,dim=1)
        return input_points, boxs

    def compute_loss(self, batch, CLASS_LOGITS, LANG_LOGITS, LOGITS, AUX_LOGITS=None):
        referential_loss = self.logit_loss(LOGITS, batch['target_pos'])
        obj_clf_loss = self.class_logits_loss(CLASS_LOGITS.transpose(2, 1), batch['class_labels'])
        lang_clf_loss = self.lang_logits_loss(LANG_LOGITS, batch['target_class'])
        total_loss = referential_loss + self.obj_cls_alpha * obj_clf_loss + self.lang_cls_alpha * lang_clf_loss
        return total_loss

    def forward(self, batch: dict, epoch=None):
        # batch['class_labels']: GT class of each obj
        # batch['target_class']：GT class of target obj
        # batch['target_pos']: GT id

        self.device = self.obj_feature_mapping[0].weight.device

        ## rotation augmentation and multi_view generation
        obj_points, boxs = self.aug_input(batch['objects'], batch['box_info'])
        B,N,P = obj_points.shape[:3]

        ## obj_encoding
        objects_features = get_siamese_features(self.object_encoder, obj_points, aggregator=torch.stack)
        
        ## obj_encoding
        obj_feats = self.obj_feature_mapping(objects_features)
        box_infos = self.box_feature_mapping(boxs)
        obj_infos = obj_feats[:, None].repeat(1, self.view_number, 1, 1) + box_infos

        # <LOSS>: obj_cls
        if self.label_lang_sup:
            label_lang_infos = self.language_encoder(**self.class_name_tokens)[0][:,0]
            CLASS_LOGITS = torch.matmul(obj_feats.reshape(B*N,-1), label_lang_infos.permute(1,0)).reshape(B,N,-1)        
        else:
            CLASS_LOGITS = self.obj_clf(obj_feats.reshape(B*N,-1)).reshape(B,N,-1)

        ## language_encoding
        lang_tokens = batch['lang_tokens']
        lang_infos = self.language_encoder(**lang_tokens)[0]

        # <LOSS>: lang_cls
        lang_features = lang_infos[:,0]
        LANG_LOGITS = self.language_clf(lang_infos[:,0])
        
        ## multi-modal_fusion
        cat_infos = obj_infos.reshape(B*self.view_number, -1, self.inner_dim)
        mem_infos = lang_infos[:, None].repeat(1, self.view_number, 1, 1).reshape(B*self.view_number, -1, self.inner_dim)
        out_feats = self.refer_encoder(cat_infos.transpose(0, 1), mem_infos.transpose(0, 1)).transpose(0, 1).reshape(B, self.view_number, -1, self.inner_dim)

        ## view_aggregation
        refer_feat = out_feats
        if self.aggregate_type=='avg':
            agg_feats = (refer_feat / self.view_number).sum(dim=1)
        elif self.aggregate_type=='avgmax':
            agg_feats = (refer_feat / self.view_number).sum(dim=1) + refer_feat.max(dim=1).values
        else:
            agg_feats = refer_feat.max(dim=1).values

        # <LOSS>: ref_cls
        LOGITS = self.object_language_clf(agg_feats).squeeze(-1)
        LOSS = self.compute_loss(batch, CLASS_LOGITS, LANG_LOGITS, LOGITS)

        return LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS
