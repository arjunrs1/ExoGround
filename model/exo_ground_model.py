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

class ExoGroundingTransformer(nn.Module):
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2,
                 use_decoder=True, 
                 sim='cos', 
                 pos_enc='learned',
                 use_text_pos_enc=0,
                 random_pos_start=1,
                 use_audio=True,
                 video_embed_dim=4096,
                 text_embed_dim=4096,
                 audio_embed_dim=2304,
                 feature_dim=512,
                 ):
        super().__init__()

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

        #initalize multi-modal encoder and narration decoder
        self.tfm_modules = []
        self.multi_modal_encoder = TemporalEncoder(
            width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.multi_modal_encoder)
        if self.use_decoder:
            self.decoder = TemporalDecoder(
            width=feature_dim, layers=num_decoder_layers, heads=8)
            self.tfm_modules.append(self.decoder)
        self.grounding_head = nn.Linear(self.feature_dim, 2)

        #initialize embeddings and projection layers
        self.video_pre_proj = nn.Linear(self.video_embed_dim, self.feature_dim, bias=False)
        self.text_pre_proj = nn.Linear(self.text_embed_dim, self.feature_dim, bias=False)
        self.ln_text_init = LayerNorm(self.feature_dim)
        self.ln_video_init = LayerNorm(self.feature_dim)
        self.ln_position_init = LayerNorm(self.feature_dim)
        self.ln_joint_post_enc = LayerNorm(self.feature_dim)

        #initialize audio embeddings and projection layers
        if self.use_audio:
            self.ln_audio_init = LayerNorm(self.feature_dim)
            self.audio_pre_proj = nn.Linear(self.audio_embed_dim, self.feature_dim, bias=False)
        
        # temporal positional encoding for video
        if self.pos_enc == 'learned':
            self.temporal_pos_embed = nn.Parameter(torch.empty(1024, self.feature_dim)) #TODO: Get rid of hardcoded 1024
            nn.init.normal_(self.temporal_pos_embed, std=0.01)
        elif self.pos_enc == 'sine':
            temporal_pos_embed = get_position_embedding_sine(self.feature_dim, 1024) #TODO: Get rid of hardcoded 1024
            self.register_buffer('temporal_pos_embed', temporal_pos_embed)

        # temporal positional encoding for text
        self.text_temporal_pos_embed = nn.Parameter(torch.empty(self.text_embed_dim, self.feature_dim))
        nn.init.normal_(self.text_temporal_pos_embed, std=0.01)

        self.mlp = nn.Linear(self.feature_dim, self.feature_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        linear_layers = [self.video_pre_proj, self.text_pre_proj, self.mlp, self.grounding_head]
        if self.use_audio:
            linear_layers.append(self.audio_pre_proj)
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
                interpolate_from=None):
        # text embedding without temporal-enc
        lang_embed_raw = self.get_textual_feature(lang_embed)

        #get audio embedding
        if audio_embed is not None:
            audio_embed_feat = self.get_audio_feature(audio_embed)
        else:
            audio_embed_feat = None

        ### Joint Encoder ###
        # get text embedding with/without temporal pos-enc
        if self.use_text_pos_enc:
            lang_embed_with_time = self.get_textual_feature_with_time(lang_embed,
                                                                       interpolate_from)
        else:
            lang_embed_with_time = lang_embed_raw

        # get multi-modal feature output from encoder   
        all_output, T = self.get_joint_feature(
            video_embed, video_padding_mask,
            lang_embed_with_time, lang_padding_mask,
            audio_embed_feat, audio_padding_mask,
            interpolate_from)

        text_features = all_output[:, :, 2*T::] if self.use_audio else all_output[:, :, T::]
        decoder_context = all_output[:, :, :2*T] if self.use_audio else all_output[:, :, :T]

        if self.use_decoder:
            decoder_output = self.decoder(x=text_features[:,-1,::].permute(1, 0, 2), memory=decoder_context[:,-1,::].permute(1, 0, 2), tgt_key_padding_mask=lang_padding_mask)
            decoder_text_features = decoder_output[-1].permute(1,0,2)
            grounding = self.grounding_head(decoder_text_features)
        else:
            # Directly use text features from encoder output for grounding
            #TODO: Need to modify grounding head in init in this case of not self.use_decoder, as it must take in a different feature size as input.
            grounding = self.grounding_head(text_features)

        output_dict = {'interval_preds': grounding}
        return output_dict

    def get_joint_feature(self, video_embed, video_padding_mask,
                          lang_embed_with_time, lang_padding_mask,
                          audio_embed=None, audio_padding_mask=None,
                          interpolate_from=None):
        """Get the joint video embedding and text embedding from the joint encoder.
        It takes both visual and textual inputs."""
        video_embed = self.ln_video_init(self.video_pre_proj(video_embed))
        B,T,_,= video_embed.shape
        if interpolate_from:
            pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
            pos_embed = F.interpolate(pos_embed_source.transpose(1,2), 
                size=T, mode='linear', align_corners=False).transpose(1,2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(T/2))
            else:
                pos_start_idx = 0
            pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx+T, :]
        
        video_embed_with_time = video_embed + self.ln_position_init(pos_embed)
        
        if audio_embed is not None:
            assert audio_embed.shape == video_embed.shape, "Audio and video inputs must match in all dimensions (batch size and timesteps and feature size)"
            audio_embed_with_time = audio_embed + self.ln_position_init(pos_embed)
            joint_embed = torch.cat((video_embed_with_time, audio_embed_with_time, lang_embed_with_time), dim=1)
        else:
            joint_embed = torch.cat((video_embed_with_time, lang_embed_with_time), dim=1)

        joint_embed = joint_embed.permute(1,0,2) # BXC -> XBC
        
        if audio_embed is not None:
            joint_padding_mask = torch.cat((video_padding_mask, audio_padding_mask, lang_padding_mask), dim=1)
        else:
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

    def get_audio_feature(self, audio_embed):
        """get audio embedding after proj and LayerNorm"""
        aud_proj = self.ln_audio_init(self.audio_pre_proj(audio_embed))
        return aud_proj


class TwinExoGroundingTransformer(nn.Module):
    """Duplicate TemporalAligner for EMA."""
    def __init__(self, m=0.999, *args, **kwargs):
        super().__init__()
        self.m = m
        self.online = ExoGroundingTransformer(*args, **kwargs)  # update by backprop
        self.target = ExoGroundingTransformer(*args, **kwargs)  # update by EMA
        self._copy_param()
        self.bert = self.online.bert
        self.get_visual_feature = self.online.get_visual_feature
        self.get_joint_feature = self.online.get_joint_feature
        self.get_textual_feature_with_time = self.online.get_textual_feature_with_time
        self.get_textual_feature = self.online.get_textual_feature
        self.get_text_visual_sim = self.online.get_text_visual_sim
        self.get_text_visual_sim_dual = self.online.get_text_visual_sim_dual
        self.get_alignability = self.online.get_alignability 

        # turn off online branch's random pos enc
        self.target.random_pos_start = 0

    def _copy_param(self):
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data.copy_(param_online.data)  # initialize
            param_target.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        '''Momentum update of the target encoder'''
        for param_online, param_target in zip(self.online.parameters(), self.target.parameters()):
            param_target.data = param_target.data * self.m + param_online.data * (1. - self.m)
    
    def forward(self, *args, **kwargs):
        return self.online(*args, **kwargs)

    @torch.no_grad()
    def forward_from_ema(self, *args, **kwargs):
        return self.target(*args, **kwargs)

