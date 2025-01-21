import torch
import torch.nn as nn
from collections import OrderedDict
from .clip_model import LayerNorm,Transformer

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class mim_decoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.embed_dim//64, batch_first=True)
        #self.cross_fushion = Transformer(width=self.embed_dim,layers=4,heads=self.embed_dim//64)

        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=4,
                                                   heads=self.embed_dim //
                                                         64)


        self.decoder_norm1 = nn.LayerNorm(self.embed_dim)
        self.decoder_norm2 = nn.LayerNorm(self.embed_dim)
        self.mim_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                        ('gelu', QuickGELU()),
                        ('ln', LayerNorm(self.embed_dim)),
                        ('fc', nn.Linear(self.embed_dim, 3*256*192))]))
        self.init_decoder_params()

        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)


    def init_decoder_params(self):
        scale = self.cross_modal_transformer.width**-0.5
        proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width)**-0.5
        nn.init.normal_(self.self_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.self_attn.out_proj.weight, std=proj_std)
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.mim_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mim_head.fc.weight, std=proj_std)

    def forward(self, image_feats, text_feats):
        image_feats = image_feats.unsqueeze(1)
        text_feats = text_feats.unsqueeze(1)

        image_feats = self.self_attn(self.ln_pre_t(image_feats),
                                     self.ln_pre_i(text_feats),
                                     self.ln_pre_i(text_feats),
                                     need_weights=False)[0]

        x = image_feats.permute(1, 0, 2)  # NLD -> LND

        #fushion_feats = self.decoder_norm1(image_feats)

        x = self.cross_modal_transformer(x)


        #x = self.cross_fushion(fushion_feats)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        # x = self.decoder_norm2(x)

        x = self.mim_head(x)

        x = x.squeeze(1)

        return x