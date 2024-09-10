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

        #initialize exo projection layer for infoNCE loss
        if self.use_distill_nce_loss or self.use_pairwise_distill_nce_loss:
            self.exo_feature_proj = nn.Linear(self.feature_dim, self.video_embed_dim)

        #initialize audio embeddings and projection layers
        if self.use_audio:
            self.ln_audio_init = LayerNorm(self.feature_dim)
            self.audio_pre_proj = nn.Linear(self.audio_embed_dim, self.feature_dim, bias=False)
        
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
                egocentric_video_embed=None,
                view_mask=None,
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

        #Get number of padded narrations
        N = lang_embed_with_time.shape[1]

        # get multi-modal feature output from encoder   
        all_output, T = self.get_joint_feature(
            video_embed, video_padding_mask,
            lang_embed_with_time, lang_padding_mask,
            audio_embed_feat, audio_padding_mask,
            interpolate_from)

        text_features = all_output[:, :, -N:]
        decoder_context = all_output[:, :, :-N]

        if self.use_distill_nce_loss and egocentric_video_embed is not None:
            exo_features = all_output[:, :, :T].mean(dim=1)
            exo_features_projected = self.exo_feature_proj(exo_features)
            distill_loss = self.compute_info_nce_loss(exo_features_projected, egocentric_video_embed)
        elif self.multi_view and self.use_pairwise_distill_nce_loss:
            exo_features = all_output[:, :, :T].mean(dim=1)
            exo_features_projected = self.exo_feature_proj(exo_features)
            distill_loss = self.compute_pairwise_info_nce_loss(exo_features_projected, view_mask=view_mask if self.pairwise_distill_mode == "all" else ~video_padding_mask)

        if self.use_decoder:
            decoder_output = self.decoder(x=text_features[:,-1,::].permute(1, 0, 2), memory=decoder_context[:,-1,::].permute(1, 0, 2), tgt_key_padding_mask=lang_padding_mask, memory_key_padding_mask=video_padding_mask)
            decoder_text_features = decoder_output[-1].permute(1,0,2)
            grounding = self.grounding_head(decoder_text_features)
        else:
            # Directly use text features from encoder output for grounding
            grounding = self.grounding_head(text_features)

        output_dict = {'interval_preds': grounding}
        if self.use_distill_nce_loss or self.use_pairwise_distill_nce_loss:
            output_dict['distill_infonce_loss'] = distill_loss

        return output_dict

    def add_positional_encoding(self, embed, interpolate_from=None):
        B, T, _ = embed.shape
        seq_len = T // self.num_max_views
        if interpolate_from:
            pos_embed_source = self.temporal_pos_embed[None, 0:interpolate_from, :]
            pos_embed = F.interpolate(pos_embed_source.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)
        else:
            if self.random_pos_start:
                pos_start_idx = np.random.randint(0, int(seq_len / 2))
            else:
                pos_start_idx = 0
            pos_embed = self.temporal_pos_embed[None, pos_start_idx:pos_start_idx + seq_len, :]
        pos_embed = pos_embed.repeat(1, self.num_max_views, 1)
        embed_with_time = embed + self.ln_position_init(pos_embed)
        return embed_with_time

    def compute_info_nce_loss(self, features, positive_features, temperature=0.1):
        """
        Compute the InfoNCE loss between features and positive features, considering both positive and negative samples.
        
        Args:
        - features (torch.Tensor): Tensor of shape (batch_size, num_features, feature_dim)
        - positive_features (torch.Tensor): Tensor of shape (batch_size, num_features, feature_dim)
        - temperature (float): A temperature scaling factor (default 0.1)
        - view_mask (torch.Tensor): Boolean tensor of shape (batch_size, num_features) indicating available views
        
        Returns:
        - torch.Tensor: Scalar tensor containing the InfoNCE loss.
        """
        assert features.size(1) == positive_features.size(1)
        # Normalize features to get unit vectors
        features_norm = F.normalize(features, p=2, dim=2)
        positive_features_norm = F.normalize(positive_features, p=2, dim=2)
        # Compute similarities
        # Transpose positive features to align with features for matrix multiplication
        similarities = torch.bmm(features_norm, positive_features_norm.transpose(1, 2)) / temperature
        # Create labels for the positive samples (diagonal elements in the batch)
        labels = torch.arange(positive_features.size(1)).to(features.device)
        # Use log-softmax for numerical stability
        log_prob = F.log_softmax(similarities, dim=2)
        # Gather the log probabilities of positive samples
        log_prob_positive = log_prob.gather(2, labels.view(1, -1).expand(features.size(0), -1).unsqueeze(2)).squeeze(2)
        # Compute the mean of the log probabilities of the positive samples
        nce_loss = -log_prob_positive.mean()
        return nce_loss

    def compute_pairwise_info_nce_loss(self, features, temperature=0.1, view_mask=None):
        """
        Compute the pairwise InfoNCE loss between all pairs of unmasked views for each item in the batch.
        
        Args:
        - features (torch.Tensor): Tensor of shape (batch_size, num_features, feature_dim)
        - temperature (float): A temperature scaling factor (default 0.1)
        - view_mask (torch.Tensor): Boolean tensor of shape (batch_size, num_features) indicating available views
        - num_splits (int): Number of splits along the feature dimension
        
        Returns:
        - torch.Tensor: Scalar tensor containing the average pairwise InfoNCE loss.
        """
        # Split the features tensor and the view mask into parts along the second dimension
        split_features = torch.chunk(features, self.num_max_views, dim=1)
        split_masks = torch.chunk(view_mask, self.num_max_views, dim=1)
        
        total_loss = 0.0
        num_valid_pairs = 0
        
        # Iterate over all pairs of feature chunks
        for i in range(self.num_max_views):
            for j in range(i + 1, self.num_max_views):
                # Compute the valid mask by logical AND operation between masks of the two views
                valid_mask = split_masks[i].squeeze(1) & split_masks[j].squeeze(1)
                
                if valid_mask.any():
                    # Select valid features for both views
                    valid_features_i = split_features[i][valid_mask]
                    valid_features_j = split_features[j][valid_mask]

                    valid_features_i = valid_features_i.unsqueeze(1)
                    valid_features_j = valid_features_j.unsqueeze(1)
                    
                    # Normalize features to get unit vectors
                    features_norm_i = F.normalize(valid_features_i, p=2, dim=2)
                    features_norm_j = F.normalize(valid_features_j, p=2, dim=2)
                    
                    # Compute similarities
                    similarities = torch.bmm(features_norm_i, features_norm_j.transpose(1, 2)) / temperature
                    
                    # Create labels for the positive samples (diagonal elements in the batch)
                    labels = torch.arange(valid_features_i.size(1), device=features.device)
                    
                    # Use log-softmax for numerical stability
                    log_prob = F.log_softmax(similarities, dim=2)
                    log_prob_positive = log_prob.gather(2, labels.view(1, -1).expand(valid_features_i.size(0), -1).unsqueeze(2)).squeeze(2)
                    
                    # Compute the mean of the log probabilities of the positive samples
                    nce_loss = -log_prob_positive.mean()
                    
                    total_loss += nce_loss
                    num_valid_pairs += 1
        
        # Average the loss across all valid pairs
        final_loss = total_loss / num_valid_pairs if num_valid_pairs > 0 else torch.tensor(0.0).to(features.device)
        return final_loss

    def get_joint_feature(self, video_embed, video_padding_mask,
                          lang_embed_with_time, lang_padding_mask,
                          audio_embed=None, audio_padding_mask=None,
                          interpolate_from=None):
        """Get the joint video embedding and text embedding from the joint encoder.
        It takes both visual and textual inputs."""
        video_embed = self.ln_video_init(self.video_pre_proj(video_embed))
        B,T,_,= video_embed.shape
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

