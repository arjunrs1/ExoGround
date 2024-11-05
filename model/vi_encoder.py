import torch
from torch import nn
import torch.nn.functional as F 
from torch.nn import LayerNorm
import numpy as np
from tfm_model import TemporalEncoder, get_position_embedding_sine

class ViewInvariantEncoder(nn.Module):
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

        #initalize video encoder
        self.tfm_modules = []
        self.video_unimodal_encoder = TemporalEncoder(
            width=feature_dim, layers=self.num_encoder_layers, heads=8)
        self.tfm_modules.append(self.video_unimodal_encoder)

        #initialize embeddings and projection layers
        self.video_pre_proj = nn.Linear(self.video_embed_dim, self.feature_dim, bias=False)
        self.ln_video_init = LayerNorm(self.feature_dim)
        self.ln_position_init = LayerNorm(self.feature_dim)
        self.ln_video_post_enc = LayerNorm(self.feature_dim)

        #initialize exo projection layer for infoNCE loss
        if self.use_distill_nce_loss or self.use_pairwise_distill_nce_loss:
            self.exo_feature_proj = nn.Linear(self.feature_dim, self.video_embed_dim)
        
        # temporal positional encoding for video
        if self.pos_enc == 'learned':
            self.temporal_pos_embed = nn.Parameter(torch.empty(1024, self.feature_dim))
            nn.init.normal_(self.temporal_pos_embed, std=0.01)
        elif self.pos_enc == 'sine':
            temporal_pos_embed = get_position_embedding_sine(self.feature_dim, 1024)
            self.register_buffer('temporal_pos_embed', temporal_pos_embed)

        self.initialize_parameters()

    def initialize_parameters(self):
        linear_layers = [self.video_pre_proj]
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

        video_encoded_features = self.get_unimodal_features("video", video_embed, video_padding_mask).mean(dim=1) #TODO: Should we do this mean pooling over heads? or max-pool? (for both vid and narr features)
        if self.use_distill_nce_loss and egocentric_video_embed is not None:
            exo_features_projected = self.exo_feature_proj(video_encoded_features)
        output_dict = {'low_dim_features': video_encoded_features, 'high_dim_features': exo_features_projected}
        return output_dict

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


class ViewInvariantMLP(nn.Module):
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

        #initalize video encoder
        self.tfm_modules = []

        self.linear_layers = []
        self.video_pre_proj = nn.Linear(self.video_embed_dim, self.video_embed_dim, bias=False)
        self.linear_layers.append(self.video_pre_proj)
        self.ln_video_init = LayerNorm(self.video_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.video_embed_dim, video_embed_dim, bias=True),
            nn.ReLU(),
            nn.Linear(video_embed_dim, self.video_embed_dim, bias=True),
        )

        self.initialize_parameters()

    def initialize_parameters(self):
        for layer in self.linear_layers:
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):  # Check if the layer is a linear layer
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

        proj_embed = self.ln_video_init(self.video_pre_proj(video_embed))

        video_encoded_features = self.mlp(proj_embed)  
        output_dict = {'low_dim_features': video_encoded_features, 'high_dim_features': video_encoded_features}
        return output_dict