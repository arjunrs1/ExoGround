import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence
from torch.nn import LayerNorm
from collections import OrderedDict
from transformers import BertModel, DistilBertModel
import numpy as np
from tfm_model import TemporalEncoder, TemporalDecoder, get_position_embedding_sine
from word2vec_model import Word2VecModel
from vi_encoder import ViewInvariantEncoder

class GroundingModel(nn.Module):
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2,
                 use_decoder=True, 
                 sim='cos', 
                 pos_enc='learned',
                 use_text_pos_enc=0,
                 random_pos_start=1,
                 use_audio=False,
                 video_embed_dim=4096,
                 text_embed_dim=4096,
                 audio_embed_dim=2304,
                 feature_dim=512,
                 use_distill_nce_loss=False,
                 multi_view=False,
                 num_max_views=1,
                 use_pairwise_distill_nce_loss=False,
                 pairwise_distill_mode="all"
                 ):
        super().__init__()

        self.view_invariant_encoder = ViewInvariantEncoder(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            use_decoder=use_decoder,
            sim=sim,
            pos_enc=pos_enc,
            use_text_pos_enc=use_text_pos_enc,
            random_pos_start=random_pos_start,
            use_audio=use_audio,
            video_embed_dim=video_embed_dim,
            text_embed_dim=text_embed_dim,
            audio_embed_dim=audio_embed_dim,
            feature_dim=feature_dim,
            use_distill_nce_loss=use_distill_nce_loss,
            multi_view=multi_view,
            num_max_views=num_max_views,
            use_pairwise_distill_nce_loss=use_pairwise_distill_nce_loss,
            pairwise_distill_mode=pairwise_distill_mode
        )
        #TODO: Initialize this with argument in command line run.sh

        #initialize args
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_decoder = use_decoder
        self.sim = sim 
        self.pos_enc = pos_enc
        self.use_text_pos_enc = use_text_pos_enc
        print(f'Use textual pos-enc in joint-encoder = {bool(use_text_pos_enc)}')
        self.random_pos_start = random_pos_start
        self.use_audio = use_audio
        self.text_embed_dim = text_embed_dim
        self.audio_embed_dim = audio_embed_dim
        self.video_embed_dim = video_embed_dim
        self.feature_dim = feature_dim
        self.use_distill_nce_loss = use_distill_nce_loss
        self.multi_view = multi_view
        self.num_max_views = num_max_views
        self.use_pairwise_distill_nce_loss = use_pairwise_distill_nce_loss
        self.pairwise_distill_mode = pairwise_distill_mode

        #initalize multi-modal encoder and narration decoder
        self.tfm_modules = []
        self.multi_modal_encoder = TemporalEncoder(
            width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.multi_modal_encoder)

        self.text_unimodal_encoder = TemporalEncoder(
            width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.text_unimodal_encoder)

        #initialize embeddings and projection layers
        self.text_pre_proj = nn.Linear(self.text_embed_dim, self.feature_dim, bias=False)
        self.ln_text_init = LayerNorm(self.feature_dim)
        self.ln_position_init = LayerNorm(self.feature_dim)
        self.ln_joint_post_enc = LayerNorm(self.feature_dim)
        self.ln_text_post_enc = LayerNorm(self.feature_dim)
        
        # temporal positional encoding for video
        if self.pos_enc == 'learned':
            self.temporal_pos_embed = nn.Parameter(torch.empty(1024, self.feature_dim))
            nn.init.normal_(self.temporal_pos_embed, std=0.01)
        elif self.pos_enc == 'sine':
            temporal_pos_embed = get_position_embedding_sine(self.feature_dim, 1024)
            self.register_buffer('temporal_pos_embed', temporal_pos_embed)

        # temporal positional encoding for text
        self.text_temporal_pos_embed = nn.Parameter(torch.empty(self.text_embed_dim, self.feature_dim))
        nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        self.mlp = nn.Linear(self.feature_dim, self.feature_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        linear_layers = [self.text_pre_proj, self.mlp]
        for layer in linear_layers:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for tfm_module in self.tfm_modules:
            proj_std = (tfm_module.width ** -0.5) * ((2 * tfm_module.layers) ** -0.5)
            attn_std = tfm_module.width ** -0.5
            fc_std = (2 * tfm_module.width) ** -0.5
            for block in tfm_module.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)


    def forward(self, video_embed, lang_embed,
                video_padding_mask, lang_padding_mask,
                audio_embed=None, audio_padding_mask=None,
                egocentric_video_embed=None,
                view_mask=None,
                interpolate_from=None):

        #get video features from view_invariant encoder
        video_output_dict = self.view_invariant_encoder(video_embed, lang_embed, video_padding_mask, lang_padding_mask, audio_embed, audio_padding_mask, egocentric_video_embed, view_mask, interpolate_from)
        video_encoded_features = video_output_dict['low_dim_features']

        # text embedding without temporal-enc
        lang_embed_raw = self.get_textual_feature(lang_embed)

        ### Joint Encoder ###
        # get text embedding with/without temporal pos-enc
        if self.use_text_pos_enc:
            lang_embed_with_time = self.get_textual_feature_with_time(lang_embed,
                                                                       interpolate_from)
        else:
            lang_embed_with_time = lang_embed_raw

        #Get number of padded narrations
        N = lang_embed_with_time.shape[1]

        text_encoded_features = self.get_unimodal_features("text", lang_embed_with_time, lang_padding_mask).mean(dim=1)

        # get multi-modal feature output from encoder   
        all_output, _ = self.get_joint_feature(
            video_encoded_features, video_padding_mask,
            text_encoded_features.squeeze(dim=1), lang_padding_mask,
            interpolate_from)

        joint_video_out = all_output[:, :, :-N]
        joint_text_out = all_output[:, :, -N:]

        # get cosine distance for Joint Encoder
        video_feature_norm_joint = joint_video_out / joint_video_out.norm(dim=-1, keepdim=True)
        text_feature_norm_joint = joint_text_out / joint_text_out.norm(dim=-1, keepdim=True)
        contrastive_logits_joint = torch.einsum("astc,bskc->astbk", 
            video_feature_norm_joint, text_feature_norm_joint)

        output_dict = {'logits_joint': contrastive_logits_joint}
        
        """ if self.return_dual_feature:
            output_dict['dual_feature_video'] = video_feature_norm
            output_dict['dual_feature_text'] = text_feature_norm
        if self.use_alignability_head:
            output_dict['dual_logits_alignability'] = self.binary_head(lang_embed_raw)
            output_dict['joint_logits_alignability'] = self.binary_head(joint_text_out) """
        return output_dict

    def get_unimodal_features(self, mode, feat_embed, padding_mask, interpolate_from=None):
        B,T,_,= feat_embed.shape
        if mode == "video":
            proj_embed = self.ln_video_init(self.video_pre_proj(feat_embed))
            seq_len = T // self.num_max_views
            if interpolate_from:
                pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
                pos_embed = F.interpolate(pos_embed_source.transpose(1,2), 
                    size=seq_len, mode='linear', align_corners=False).transpose(1,2)
            else:
                if self.random_pos_start:
                    pos_start_idx = np.random.randint(0, int(seq_len/2))
                else:
                    pos_start_idx = 0
                pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+seq_len, :]
            pos_embed = pos_embed.repeat(1, self.num_max_views, 1)
            feat_embed_with_time = proj_embed + self.ln_position_init(pos_embed)
        else:
            feat_embed_with_time = feat_embed
        feat_embed_with_time = feat_embed_with_time.permute(1,0,2) # BXC -> XBC
        if mode == "video":    
            feat_output = self.video_unimodal_encoder(feat_embed_with_time, padding_mask)
            feat_output[-1] = self.ln_video_post_enc(feat_output[-1])
        else:
            feat_output = self.text_unimodal_encoder(feat_embed_with_time, padding_mask)
            feat_output[-1] =self.ln_text_post_enc(feat_output[-1])
        feat_output = torch.stack(feat_output, dim=1).permute(2,1,0,3)  # B,Stage,X,C
        return feat_output

    def get_joint_feature(self, video_embed_with_time, video_padding_mask,
                          lang_embed_with_time, lang_padding_mask,
                          interpolate_from=None):
        """Get the joint video embedding and text embedding from the joint encoder.
        It takes both visual and textual inputs."""
        B,T,_,= video_embed_with_time.shape

        joint_embed = torch.cat((video_embed_with_time, lang_embed_with_time), dim=1)
        joint_embed = joint_embed.permute(1,0,2) # BXC -> XBC
        joint_padding_mask = torch.cat((video_padding_mask, lang_padding_mask), dim=1)

        joint_output = self.multi_modal_encoder(joint_embed, joint_padding_mask)
        joint_output[-1] = self.ln_joint_post_enc(joint_output[-1])

        joint_output = torch.stack(joint_output, dim=1).permute(2,1,0,3)  # B,Stage,X,C
        return joint_output, T


    def get_textual_feature_with_time(self, lang_embed, interpolate_from=None):
        """add proper positional embedding to text
        lang_embed: tensor [B,N,C]"""
        text_proj = self.ln_text_init(self.text_pre_proj(lang_embed))
        N = lang_embed.shape[1]
        if interpolate_from:
            text_pos_embed_source = self.text_temporal_pos_embed[None, 0:interpolate_from, :]
            text_pos_embed = F.interpolate(text_pos_embed_source.transpose(1,2), 
                size=N, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(N/2))
            else:
                pos_start_idx = 0
            text_pos_embed = self.text_temporal_pos_embed[None, pos_start_idx:pos_start_idx+N, :]
        return text_proj + self.ln_position_init(text_pos_embed)


    def get_textual_feature(self, lang_embed):
        """get text embedding after proj and LayerNorm"""
        text_proj = self.ln_text_init(self.text_pre_proj(lang_embed))
        return text_proj

