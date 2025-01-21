from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from model.ot import optimal_transport_dist
from model import mim_blocks
import torch.nn.functional as F


class TIRRS(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.citc_lambda1 = args.citc_lambda1
        self.citc_lambda2 = args.citc_lambda2
        self.citc_gamma = args.citc_gamma
        self.ot_gamma = args.ot_gamma
        self.dl_gamma = args.dl_gamma
        self.aug_ss = args.aug_ss
        self.ss_loss_gamma = args.ss_loss_gamma

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.check_model, args.pretrain_choice,
                                                                      args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mim' in args.loss_names:
            self.mask_token = nn.Parameter(torch.zeros([1, 3, 32, 32]))
            self.mim_gen = mim_blocks.mim_decoder(self.embed_dim)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def build_masks_for_one_batch(self, batch_size, mask_ratio=0.75, patch_num=64):
        mask_length = int(patch_num * mask_ratio)
        mask_batch = []
        for i in range(int(batch_size)):
            mask_idx = torch.randperm(patch_num)[:mask_length]
            mask1 = torch.zeros([patch_num])
            mask1[mask_idx] = 1
            mask_batch.append(mask1)
        mask = torch.stack(mask_batch, dim=0)
        return mask

    def build_masked_image(self, image, masks):
        assert masks is not None
        image = image.cuda()
        masks = masks.cuda()
        B, C, H, W = image.shape
        mask_tokens = self.mask_token.repeat(B, 1, 8, 8)
        temp_mask = masks.reshape(B, 8, 8).unsqueeze(1).repeat(1, 3, 32, 32)
        x = torch.mul(image, (1.0 - temp_mask)) + torch.mul(temp_mask, mask_tokens)
        return x

    def get_unmasked_image(self, image, masks):
        image = image.cuda()
        masks = masks.cuda()
        B, C, H, W = image.shape
        temp_mask = masks.reshape(B, 8, 8).unsqueeze(1).repeat(1, 3, 32, 32)
        reserve_token = (temp_mask == 1)
        image = image[reserve_token]
        return image

    def get_mim_loss(self, recon, img):
        # l1 = nn.L1Loss()
        recon = recon.float()
        img = img.float()

        l1 = nn.L1Loss(reduction='mean')
        return l1(recon, img)

    def _build_mlp(self, in_dim=512, mlp_dim=128, out_dim=512):
        return nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim)
        )

    def forward(self, batch, epoch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        if self.aug_ss:
            images_aug_ss_1 = batch['aug_ss_1']
            images_aug_ss_2 = batch['aug_ss_2']

        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'aug_ss' in self.current_task:
            simclr_mlp = self._build_mlp(in_dim=512, mlp_dim=128, out_dim=512).to(images_aug_ss_1.device)

            images_aug_ss_1_feats = self.base_model.encode_image(images_aug_ss_1)
            images_aug_ss_2_feats = self.base_model.encode_image(images_aug_ss_2)
            aug_ss_1_feats = images_aug_ss_1_feats[:, 0, :].float()
            aug_ss_2_feats = images_aug_ss_2_feats[:, 0, :].float()

            aug1_embed = simclr_mlp(aug_ss_1_feats)
            aug2_embed = simclr_mlp(aug_ss_2_feats)

            q_a = F.normalize(aug1_embed, dim=-1, p=2)
            q_b = F.normalize(aug2_embed, dim=-1, p=2)
            local_batch_size = q_a.size(0)
            labels = torch.arange(local_batch_size, device=q_a.device)
            ss_loss = objectives.compute_simclr_loss(q_a, q_b, q_a, q_b, labels, 0.1)
            ss_loss = ss_loss * self.ss_loss_gamma
            ret.update({'ss_loss': ss_loss})

        if 'ssimg_cau' in self.current_task:
            aug_images = batch['aug_images']
            aug_image_feats = self.base_model.encode_image(aug_images)
            aug_i_feats = aug_image_feats[:, 0, :].float()
            sscau_loss = objectives.compute_sscau_loss(i_feats,aug_i_feats)
            sscau_loss = sscau_loss * self.args.ss_loss_gamma
            ret.update({'ss_loss': sscau_loss})

        if 'ssimg' in self.current_task:
            aug_images = batch['aug_images']
            aug_image_feats = self.base_model.encode_image(aug_images)
            aug_i_feats = aug_image_feats[:, 0, :].float()
            ss_loss = objectives.compute_ssimg_loss(aug_i_feats, i_feats, logit_scale)
            ss_loss = ss_loss * self.args.ss_loss_gamma
            ret.update({'ss_loss': ss_loss})

        if 'dl' in self.current_task:
            dl_loss = objectives.get_unc_loss(i_feats, t_feats, epoch, self.args.num_epoch, amplititude=0.7)
            dl_loss = dl_loss * self.dl_gamma
            ret.update({'dl_loss':dl_loss})


        if 'cyclic' in self.current_task:
            cyclic_loss = objectives.get_cyclic_loss_one(i_feats, t_feats, logit_scale,self.citc_lambda1,self.citc_lambda2)
            cyclic_loss = cyclic_loss * self.citc_gamma

            # cyclic_loss = objectives.get_cyclic_loss_two(i_feats, t_feats, logit_scale,self.citc_gamma)

            ret.update({'cyclic_loss': cyclic_loss})

        if 'ot' in self.current_task:
            ot_loss = optimal_transport_dist(text_feats.float(), image_feats.float())
            ot_loss = ot_loss * self.ot_gamma
            ret.update({'ot_loss': ot_loss})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        if 'mim' in self.current_task:
            mask_for_one_batch = self.build_masks_for_one_batch(images.shape[0])
            masked_img = self.build_masked_image(images, mask_for_one_batch)
            masked_image_feats = self.base_model.encode_image(masked_img)
            masked_i_feats = masked_image_feats[:, 0, :]
            mask_t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]
            recon_image = self.mim_gen(masked_i_feats, mask_t_feats)
            temp_image = self.get_unmasked_image(images, mask_for_one_batch)

            temp_image = temp_image.reshape([images.shape[0], 3 * 256 * 192])
            mim_loss = self.get_mim_loss(recon_image, temp_image)

            ret.update({'mim_loss': mim_loss * self.args.mim_loss_weight})

        return ret


def build_model(args, num_classes=11003):
    model = TIRRS(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
