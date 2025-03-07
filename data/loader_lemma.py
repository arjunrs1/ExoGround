import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from collections import Counter
import os
import json
import ast
import random
import itertools
from tqdm import tqdm
import torch.nn.functional as F

class LemmaDataLoader(Dataset):
    def __init__(self,
                split='train',
                duration=64,
                hop_length=5,
                views="all",
                use_distill_nce_loss=False,
                curriculum_train=False,
                same_view_negative=False,
                use_tf_video_features=False,
                reverse_ranking=False,
                randomize_ranking=False,
                fps=24):

        #Assign params:        
        self.split = split
        self.duration = duration
        self.hop_length = hop_length
        self.views = views
        self.use_distill_nce_loss = use_distill_nce_loss
        self.curriculum_train = curriculum_train
        self.same_view_negative = same_view_negative
        self.use_tf_video_features = use_tf_video_features
        self.reverse_ranking = reverse_ranking
        self.randomize_ranking = randomize_ranking
        self.fps = fps

        #Define paths:
        self.base_path = '/scratch/projects/CCR24058/lemma_dataset'
        self.vid_feat_rel_path = "vid_feats_1_fps_real"
        self.split_path = f"/work/10323/asomaya1/vista/code/exo_narration_grounding/splits/lemma_splits/{split}.csv"
        self.keysteps_annotations_path = f'/work/10323/asomaya1/vista/code/exo_narration_grounding/data_processing/time_interval_annotation_files/lemma/keystep_annotations/{split}.csv'
        self.hoi_metadata_path = "/work/10323/asomaya1/vista/code/exo_narration_grounding/data_processing/time_interval_annotation_files/lemma/keystep_annotations/all.csv"

        assert not ((self.views == "ego") and (self.use_distill_nce_loss)) #We cannot train on ego only and distill from ego view simultaneously

        if self.curriculum_train:
            assert self.split == "train"

        self.annotations = pd.read_csv(self.keysteps_annotations_path)
        self.split_data = pd.read_csv(self.split_path)
        if not self.use_tf_video_features:
            self.video_feature_path = os.path.join(self.base_path, self.vid_feat_rel_path)
        else:
            self.video_feature_path = "/checkpoint/arjunrs1/vi_encoder_features/egoexo4d_features_REAL"

        self.narration_feature_path = os.path.join(self.base_path, "keystep_feats")
        self.current_phase = 0

        hoi_metadata = pd.read_csv(self.hoi_metadata_path)
        self.hoi_idx_to_text_map = dict(zip(hoi_metadata["hoi_index"], hoi_metadata["natural_language"]))

        self.unique_narr_id_to_hoi_idx_map = {
            row["unique_narration_id"]: row["narration"].removeprefix("HOI ")
            for _, row in self.annotations.iterrows()
        }

        self.window_csv_path = os.path.join(self.base_path, f'joint_{split}_{views}_ks=True_ct={curriculum_train}_exos=all_windows_dur={duration}_hop={hop_length}.csv')
        self.precompute_windows()

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def collate_fn(batch):
        # Exclude 'metadata' from default collation
        exclusions = ['metadata']
        batch_rest = [{k: v for k, v in item.items() if k not in exclusions} for item in batch]
        collated_data = default_collate(batch_rest)
        if 'metadata' in exclusions:
            metadata_keys = batch[0]['metadata'].keys()  # Assuming all items have the same keys
            collated_metadata = {}
            for key in metadata_keys:
                collated_metadata[key] = [item['metadata'][key] for item in batch]
            collated_data['metadata'] = collated_metadata
        return collated_data

    def set_phase(self, phase):
        print(f"ENTERING PHASE: {phase}")
        self.current_phase = phase

    def precompute_windows(self):
        if not os.path.exists(self.window_csv_path):
            print("Computing windows...")
            windows = []
            for _, row in tqdm(self.split_data.iterrows(), total=len(self.split_data)):
                video_id = row['video_id']
                duration_sec = int(row['duration_sec'])
                exo_cam = "master"
                ego_cam = "fpv1"

                max_start_sec = int(duration_sec) - self.duration
                for start_sec in range(0, max_start_sec + 1, self.hop_length):
                    end_sec = start_sec + self.duration
                    narrations = self.annotations[
                        (self.annotations['vid_name'] == video_id) &
                        (self.annotations['start_frame'] / self.fps <= end_sec) &
                        (self.annotations['end_frame'] / self.fps >= start_sec)
                    ]
                    print(video_id)
                    print(start_sec)
                    print(end_sec)
                    print(narrations)
                    for _, row in narrations.iterrows():
                        print(os.path.join(self.narration_feature_path, f"{self.unique_narr_id_to_hoi_idx_map[row['unique_narration_id']]}.pt"))
                    narration_ids = [row['unique_narration_id'] for _, row in narrations.iterrows() if os.path.exists(os.path.join(self.narration_feature_path, f"{self.unique_narr_id_to_hoi_idx_map[row['unique_narration_id']]}.pt"))]
                    if len(narrations) != 0:
                        windows.append([video_id, exo_cam, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                        if self.split == "test":
                            windows.append([video_id, ego_cam, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                                     
            columns = ['video_id', 'exo_cam', 'ego_cam', 'start_sec', 'end_sec', 'narration_ids']
            windows_df = pd.DataFrame(windows, columns=columns)
            windows_df.to_csv(self.window_csv_path, index=False)
            self.windows = windows_df
        else:
            print("Loading windows...")
            self.windows = pd.read_csv(self.window_csv_path)
        print(f"Number of {self.split} windows:")
        print(len(self.windows))


    def create_valid_views_mask(self, exo_video_feats, pos_views, neg_views):
        max_views, seq_len, feat_dim = exo_video_feats.shape
        seq_mask = torch.zeros((max_views, seq_len), dtype=torch.bool)
        true_values = torch.ones(seq_len, dtype=torch.bool)
        seq_mask.scatter_(0, pos_views.unsqueeze(0), true_values.unsqueeze(0))
        return seq_mask

    def get_exo_features_and_target(self, video_id, ego_cam, exo_cam, start_sec, end_sec):
        
        exo_video_feats = []
        exo_video_feats.append(torch.load(os.path.join(self.video_feature_path, video_id, ego_cam, f"{video_id}_{ego_cam}_combined.pt"))[start_sec:end_sec])
        exo_video_feats.append(torch.load(os.path.join(self.video_feature_path, video_id, exo_cam, f"{video_id}_{exo_cam}_combined.pt"))[start_sec:end_sec])
        
        exo_video_feats = torch.stack(exo_video_feats, dim=0)
        
        if self.split == "test" and exo_cam != "master":
            per_second_views = ["0" for _ in range(self.duration)]
        else:
            per_second_views = ["1" for _ in range(self.duration)] 
        target_indices = torch.zeros(self.duration, dtype=torch.long)
        neg_indices = torch.ones(self.duration, dtype=torch.long)
        
        valid_views_mask = self.create_valid_views_mask(exo_video_feats, pos_views=target_indices, neg_views=neg_indices)
        return exo_video_feats, target_indices, neg_indices, valid_views_mask, per_second_views

    def get_same_view_neg_idxs(self, ego_vid_features, narration_features, unnorm_starts, unnorm_ends):
        same_view_neg_indices = []
        if len(narration_features) == 1:
            # Calculate the relative start and end
            rel_start = int(max(0, unnorm_starts[0]))
            rel_end = int(min(self.duration - 1, unnorm_ends[0]))
            for i in range(ego_vid_features.size(0)):
                if rel_start <= i <= rel_end:
                    # Current index is within the narration interval, choose outside
                    if rel_start > 0 and rel_end < self.duration - 1:
                        neg_idx = random.choice(list(range(0, rel_start)) + list(range(rel_end + 1, self.duration)))
                    elif rel_start > 0:
                        neg_idx = random.randint(0, rel_start - 1)
                    elif rel_end < self.duration - 1:
                        neg_idx = random.randint(rel_end + 1, self.duration - 1)
                    else:
                        neg_idx = random.randint(0, self.duration - 1)  # Fallback if no valid negative
                else:
                    # Current index is outside the narration interval, choose inside
                    neg_idx = random.randint(rel_start, rel_end)
                same_view_neg_indices.append(neg_idx)
        else:
            device = ego_vid_features.device
            narration_features_tensor = torch.stack(narration_features).squeeze(1).to(device)
            similarity_matrix = torch.mm(ego_vid_features, narration_features_tensor.t())
            similarity_matrix = similarity_matrix / (
                    ego_vid_features.norm(dim=1, keepdim=True) * narration_features_tensor.norm(dim=1).t()
                )
            #Find the index of the least similar narration for each video feature
            least_sim_indices = similarity_matrix.argmin(dim=1)
            for i, least_sim_idx in enumerate(least_sim_indices):
                # Get the relative start and end for this narration
                rel_start = int(max(0, unnorm_starts[least_sim_idx]))
                rel_end = int(min(self.duration-1, unnorm_ends[least_sim_idx]))
                # Randomly select an index within this interval
                if rel_start <= rel_end:
                    neg_idx = random.randint(rel_start, rel_end)
                else:
                    neg_idx = random.randint(0, self.duration-1)  # Fallback if interval is invalid
                same_view_neg_indices.append(neg_idx)
        return torch.tensor(same_view_neg_indices, dtype=torch.long)

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        video_id, exo_cam, ego_cam, start_sec, end_sec = window['video_id'], window['exo_cam'], window['ego_cam'], window['start_sec'], window['end_sec']
        narration_ids = window['narration_ids'].split(',')
        
        # Load video features
        video_features_list = []
        features = torch.load(os.path.join(self.video_feature_path, video_id, exo_cam, f"{video_id}_{exo_cam}_combined.pt"))[start_sec:end_sec]
        video_features_list.append(features)

        video_features = torch.cat(video_features_list, dim=0)
        exo_video_features, target, neg_target, valid_views_mask, per_second_views = self.get_exo_features_and_target(video_id, ego_cam, exo_cam, start_sec, end_sec)
        
        # Load narration features
        narration_features = []
        for nid in narration_ids:
            try:
                hoi_idx = self.unique_narr_id_to_hoi_idx_map[nid]
                feature = torch.load(os.path.join(self.narration_feature_path, f"{hoi_idx}.pt"))
                narration_features.append(feature)
            except:
                print(f"Bad narration: {nid}")

        # Load metadata
        narrations = self.annotations[self.annotations['unique_narration_id'].isin(narration_ids)]
        narration_texts, starts, ends, unnorm_starts, unnorm_ends = [], [], [], [], []
        for _, row in narrations.iterrows():
            narration_texts.append(self.hoi_idx_to_text_map[int(row['narration'].split(" ")[-1])])
            sec_start = (int(row['start_frame']) / self.fps) - start_sec
            sec_end = (int(row['end_frame']) / self.fps) - start_sec
            unnorm_starts.append(sec_start)
            unnorm_ends.append(sec_end)
            norm_start = max(sec_start / self.duration, 0.0)
            norm_end = min(sec_end / self.duration, 1.0)
            starts.append(norm_start)
            ends.append(norm_end)

        #truncate the narrations if too many:
        narration_features = narration_features[:self.duration]
        narration_texts = narration_texts[:self.duration]
        unnorm_starts = unnorm_starts[:self.duration]
        unnorm_ends = unnorm_ends[:self.duration]
        starts = starts[:self.duration]
        ends = ends[:self.duration]

        if self.same_view_negative:
            #Load the exo video features
            exo_features_vid_path = os.path.join(self.video_feature_path, video_id, exo_cam, f"{video_id}_{exo_cam}_combined.pt")
            exo_vid_features = torch.load(exo_features_vid_path)[start_sec:end_sec]
            same_view_neg_indxs = self.get_same_view_neg_idxs(exo_vid_features, 
                                                              narration_features,
                                                              unnorm_starts,
                                                              unnorm_ends)

        padded_narration_features = torch.zeros(int(self.duration), 1, 4096) 
        padded_starts = torch.zeros(int(self.duration),1)
        padded_ends = torch.zeros(int(self.duration),1)
        narration_padding_mask = torch.ones(int(self.duration), dtype=torch.bool)

        if narration_features:
            padded_starts[:len(narration_features)] = torch.tensor(starts).unsqueeze(1)
            padded_ends[:len(narration_features)] = torch.tensor(ends).unsqueeze(1)
            padded_narration_features[:len(narration_features),::] = torch.stack(narration_features)
            narration_padding_mask[:len(narration_features)] = 0

        metadata = {"narrations": narration_texts, 
                    "video_id": video_id, 
                    "exo_camera": exo_cam, 
                    "start_sec": start_sec,
                    "per_second_views": per_second_views}

        output_dict = {
            'video_features': video_features.squeeze(1),
            'video_padding_mask': torch.zeros(video_features.size(0), dtype=torch.bool),
            'narration_features': padded_narration_features.squeeze(1),
            'narration_padding_mask': narration_padding_mask,
            'starts': padded_starts.squeeze(1),
            'ends': padded_ends.squeeze(1),
            'metadata' : metadata
        }
        
        if self.use_distill_nce_loss:
            output_dict['ego_video_features'] = exo_video_features.squeeze(1)
            output_dict['view_rank_label'] = target
            output_dict['view_rank_neg_label'] = neg_target
            output_dict['valid_views_mask'] = valid_views_mask

        output_dict['mean'] = (output_dict['starts'] + output_dict['ends']) / 2
        output_dict['duration'] = torch.abs(output_dict['ends']-output_dict['starts'])

        if self.same_view_negative:
            output_dict['same_view_neg_idxs'] = same_view_neg_indxs

        return output_dict