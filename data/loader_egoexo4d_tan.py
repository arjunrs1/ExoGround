import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import json
import ast
import random
import itertools
from tqdm import tqdm
import torch.nn.functional as F
import sys
sys.path.append('../model')
from word2vec_model import Word2VecTokenizer

class EgoExo4DDataLoaderTAN(Dataset):
    def __init__(self,
                split='train',
                duration=20,
                hop_length=10,
                use_audio=False,
                use_keysteps=False,
                views="exo",
                use_distill_nce_loss=False,
                use_center_duration=True,
                multi_view_single_exo_inference=False,
                multi_view_egoexo=False,
                num_max_views=None,
                randomize_narration_order=False,
                curriculum_train=False,
                sorted_curr_train="sorted",
                stitched_best_exo_distill=False,
                model="joint",
                exo_mode="all",
                minimum_four_exo_takes=False,
                same_view_negative=False,
                use_tf_video_features=False,
                tokenizer=None,
                fps=30):
        self.split = split
        self.duration = duration
        self.hop_length = hop_length
        self.use_audio = use_audio
        self.use_keysteps = use_keysteps
        self.views = views
        self.multi_view = self.views == "multi"
        self.use_distill_nce_loss = use_distill_nce_loss
        self.use_center_duration = use_center_duration
        self.multi_view_single_exo_inference = multi_view_single_exo_inference
        self.multi_view_egoexo = multi_view_egoexo
        self.num_max_views = num_max_views
        self.randomize_narration_order = randomize_narration_order
        self.curriculum_train = curriculum_train
        self.sorted_curr_train = sorted_curr_train
        self.stitched_best_exo_distill = stitched_best_exo_distill
        self.model = model
        self.exo_mode = exo_mode
        self.minimum_four_exo_takes = minimum_four_exo_takes
        self.same_view_negative = same_view_negative
        self.use_tf_video_features = use_tf_video_features
        self.tokenizer = tokenizer
        self.fps = fps
        self.base_path = '/private/home/arjunrs1/egoexo4d_features'
        self.vid_feat_rel_path = "checkpoint/yalesong/EgoExo4D_EgoVLPv2_arxivC/extracted_features_egovlp2/EgoVLPv2_Pretraining_bs512-lr3e_5-Ep20"
        self.split_path = f"/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/{split}.csv"
        self.annotation_path = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/narration_annotations/{split}.csv'
        self.keysteps_annotations_path = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/{split}.csv'
        self.takes_path = "/datasets01/egoexo4d/v2/takes/"
        self.camera_pose_train_path = "/datasets01/egoexo4d/v2/annotations/ego_pose/train/camera_pose"
        self.camera_pose_val_path = "/datasets01/egoexo4d/v2/annotations/ego_pose/val/camera_pose"
        self.camera_pose_test_path = "/datasets01/egoexo4d/v2/annotations/ego_pose/test/camera_pose"
        self.camera_rankings_path = os.path.join(self.base_path, "all_camera_rankings.json")
        self.best_exo_annotations_path = os.path.join(self.base_path, "best_exo_annotations.json")

        self.take_uid_cam_pose_split_map = {}
        for camera_path in [self.camera_pose_train_path, self.camera_pose_val_path, self.camera_pose_test_path]:
            for cam_file in os.listdir(camera_path):
                self.take_uid_cam_pose_split_map[cam_file.split(".")[0]] = camera_path.split("/")[-2]

        self.atomic_take_cam_map_train_path = f'/datasets01/egoexo4d/v2/annotations/atomic_descriptions_train.json'
        self.atomic_take_cam_map_test_path = f'/datasets01/egoexo4d/v2/annotations/atomic_descriptions_val.json'

        with open(self.atomic_take_cam_map_train_path, "rb") as f:
            atomic_descriptions_train_data = json.load(f)
            self.atomic_take_cam_map_train = atomic_descriptions_train_data['take_cam_id_map']
            self.atomic_descriptions_train = atomic_descriptions_train_data['annotations']

        with open(self.atomic_take_cam_map_test_path, "rb") as f:
            self.atomic_take_cam_map_test = json.load(f)['take_cam_id_map']

        with open(self.camera_rankings_path, "rb") as f:
            self.camera_rankings = json.load(f)

        assert not ((self.views == "ego") and (self.use_distill_nce_loss)) #We cannot train on ego only and distill from ego view simultaneously

        if self.split != "train":
            assert self.exo_mode == "all" # for val/testing, ensure we are using all views

        if self.use_keysteps:
            self.annotations = pd.read_csv(self.keysteps_annotations_path)
        else:
            self.annotations = pd.read_csv(self.annotation_path)
        self.split_data = pd.read_csv(self.split_path)
        self.video_feature_path = os.path.join(self.base_path, self.vid_feat_rel_path)
        self.audio_feature_path = os.path.join(self.base_path, 'audio_features', f'{split}')
        self.narration_feature_path = os.path.join(self.base_path, 'narration_features')
        if self.use_keysteps:
            self.narration_feature_path = os.path.join(self.narration_feature_path, "keystep_features")
        self.current_phase = 0
        
        self.windows_path = self.base_path
        if self.minimum_four_exo_takes:
            self.windows_path = os.path.join(self.windows_path, "mismatched_removed")

        model_prepend_str = "grounding" if self.model in ['grounding', 'joint'] else "view_invariant"
        self.window_csv_path = os.path.join(self.windows_path, f'{model_prepend_str}_{split}_{views}_ks={use_keysteps}_ct={curriculum_train}_exos={exo_mode}_windows.csv')
        self.precompute_windows()

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def collate_fn(batch):
        # Exclude 'metadata' from default collation
        exclusions = ['metadata', 'start', 'end']
        batch_rest = [{k: v for k, v in item.items() if k not in exclusions} for item in batch]
        collated_data = default_collate(batch_rest)
        if 'metadata' in exclusions:
            metadata_keys = batch[0]['metadata'].keys()  # Assuming all items have the same keys
            collated_metadata = {}
            for key in metadata_keys:
                collated_metadata[key] = [item['metadata'][key] for item in batch]
            collated_data['metadata'] = collated_metadata
        if 'start' in exclusions:
            collated_data['start'] = [item['start'] for item in batch]
        if 'end' in exclusions:    
            collated_data['end'] = [item['end'] for item in batch]
        return collated_data

    def set_phase(self, phase):
        print(f"ENTERING PHASE: {phase}")
        self.current_phase = phase

    def precompute_windows(self):
        if not os.path.exists(self.window_csv_path):
            print("Computing windows...")
            windows = []
            if self.split == "train" and self.views == "all":
                cam_dists = []
            for _, row in tqdm(self.split_data.iterrows(), total=len(self.split_data)):
                video_id = row['take_name']
                take_uid = row['take_uid']
                duration_sec = int(row['duration_sec'])
                exo_cams = [cam.split(".")[0] for cam in os.listdir(os.path.join(self.takes_path, video_id, "frame_aligned_videos")) if (".mp4" in cam.lower()) and ("aria" not in cam.lower())]
                ego_cam = row['ego_camera_path'].split("/")[-1].split(".")[0]
                cams = exo_cams if self.views == "exo" else ([ego_cam] if self.views == "ego" else [ego_cam] + exo_cams)

                max_start_sec = int(duration_sec) - self.duration
                for start_sec in range(0, max_start_sec + 1, self.hop_length):
                    end_sec = start_sec + self.duration
                    narrations = self.annotations[
                        (self.annotations['take_uid'] == video_id) &
                        (self.annotations['start_frame'] / self.fps <= end_sec) &
                        (self.annotations['end_frame'] / self.fps >= start_sec)
                    ]
                    narration_ids = [row['unique_narration_id'] for _, row in narrations.iterrows() if os.path.exists(os.path.join(self.narration_feature_path, video_id, f"{row['unique_narration_id']}.pt"))]
                    if len(narrations) != 0:
                        if self.multi_view:
                            windows.append([video_id, cams if self.multi_view_egoexo else exo_cams, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                        else: #all view
                            if self.curriculum_train:
                                #TODO: We are not doing this anymore - curr training will be implemented separately. This 
                                #Can be reverted to the same as for val/test (see  else statement below)
                                sorted_cams, cam_distances = self.camera_view_order(take_uid, cams, start_sec, end_sec, ego_cam)
                                far_close_pairs = list(itertools.combinations(sorted_cams, 2))
                                for cam1, cam2 in far_close_pairs:
                                    windows.append([video_id, cam1, cam2, start_sec, end_sec, ','.join(narration_ids)])
                                    cam_dists.append(cam_distances[cam1])
                                if ego_cam in cams:
                                    windows.append([video_id, ego_cam, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                                    cam_dists.append(0)
                            else: 
                                #for val/test set, we should only include each view once (and only exos)
                                for camera in exo_cams:
                                    windows.append([video_id, camera, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                                     
            columns = ['video_id', 'exo_cam', 'ego_cam', 'start_sec', 'end_sec', 'narration_ids']
            windows_df = pd.DataFrame(windows, columns=columns)
            windows_df.to_csv(self.window_csv_path, index=False)
            self.windows = windows_df
            if self.curriculum_train:
                cam_distances_df = pd.DataFrame(cam_dists, columns=["cam_ego_distance"])
                cam_distances_df.to_csv(self.windows_cam_distances_path, index=False)
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

    def get_exo_features_and_target(self, video_id, ego_cam, exo_cam, take_ego_id, start_sec, end_sec):
        take_uid = self.split_data[self.split_data['take_name'] == video_id]['take_uid'].iloc[0]
        exo_cams = ['ego'] + [cam.split(".")[0] for cam in os.listdir(os.path.join(self.takes_path, video_id, "frame_aligned_videos")) if (".mp4" in cam.lower()) and ("aria" not in cam.lower())]
        
        if ego_cam != exo_cam:
            exo_cams.remove(exo_cam)
        exo_video_feats = []
        exo_video_feats.append(torch.load(os.path.join(self.video_feature_path, f"{take_ego_id}.pt"))[start_sec:end_sec])
        
        for exo_c in exo_cams[1:]:
            take_exo_id = f"{video_id}_{exo_c}"
            features = torch.load(os.path.join(self.video_feature_path, f"{take_exo_id}.pt"))[start_sec:end_sec]
            exo_video_feats.append(features)
        exo_video_feats = torch.stack(exo_video_feats, dim=0)
        current_N = exo_video_feats.shape[0]
        max_views = 7
        
        if current_N < max_views:
            pad_size = max_views - current_N
            exo_video_feats = F.pad(exo_video_feats, (0, 0, 0, 0, 0, pad_size), "constant", 0)
        
        target = self.camera_rankings[take_uid]
        target_indices = torch.zeros(self.duration, dtype=torch.long)
        neg_indices = torch.zeros(self.duration, dtype=torch.long)
        per_second_views = [] 
        
        for t in range(start_sec, end_sec):
            tth_second_rank = target[str(t)]
            assert tth_second_rank is not None
            curr_view_rank = "ego" if ego_cam == exo_cam else self.find_key_by_value(tth_second_rank, exo_cam)
            per_second_views.append(curr_view_rank)
            if tth_second_rank:
                if curr_view_rank in ['ego', 'unk']:
                     best_view = tth_second_rank["0"]
                else:
                    if self.curriculum_train and self.sorted_curr_train in ['phased']:
                        best_view_rank = max(0, int(curr_view_rank)-(self.current_phase+1)) if int(curr_view_rank) != 0 else -1
                        #best_view_rank = max(-1, int(curr_view_rank)-(self.current_phase+1)) #NOTE: uncomment this line to switch to Ego-best (and below too)
                    else:
                        best_view_rank = 0 if int(curr_view_rank) != 0 else -1
                        # best_view_rank = -1 #NOTE: uncomment this line to switch to Ego-best (and above too)
                    best_view = "ego" if best_view_rank == -1 else tth_second_rank[str(best_view_rank)]
                best_view_idx = exo_cams.index(best_view)
                worst_view_rank = np.array([int(v) for v in list(tth_second_rank.keys())]).max()
                if curr_view_rank == str(worst_view_rank):
                    worst_view_rank -= 1 #TODO: Fix this, for rank 4 this is using rank 3 as neg (wrong)
                worst_view = tth_second_rank[str(worst_view_rank)]
                worst_view_idx = exo_cams.index(worst_view) 
                #assert worst_view_idx != best_view_idx #TODO: add this back in later on!!
            target_indices[t-start_sec] = best_view_idx
            neg_indices[t-start_sec] = worst_view_idx
        
        valid_views_mask = self.create_valid_views_mask(exo_video_feats, pos_views=target_indices, neg_views=neg_indices)
        return exo_video_feats, target_indices, neg_indices, valid_views_mask, per_second_views

    def find_key_by_value(self, data, search_value):
        if data:
            for key, value in data.items():
                if value == search_value:
                    return key
        return "unk" #TODO: Can move this function to utils

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        video_id, exo_cams, ego_cam, start_sec, end_sec = window['video_id'], window['exo_cam'], window['ego_cam'], window['start_sec'], window['end_sec']
        take_ego_id = f"{video_id}_{ego_cam}"
        narration_ids = window['narration_ids'].split(',')
        exo_cams = [exo_cams]
        
        # Load video features
        video_features_list = []
        for exo_cam in exo_cams:
            take_exo_id = f"{video_id}_{exo_cam}"
            features = torch.load(os.path.join(self.video_feature_path, f"{take_exo_id}.pt"))[start_sec:end_sec]
            video_features_list.append(features)

        video_features = torch.cat(video_features_list, dim=0)
        #exo_video_features, target, neg_target, valid_views_mask, per_second_views = self.get_exo_features_and_target(video_id, ego_cam, exo_cams[0], take_ego_id, start_sec, end_sec)
        
        # Load narration features
        narration_features = []
        for nid in narration_ids:
            try:
                feature = torch.load(os.path.join(self.narration_feature_path, video_id, f"{nid}.pt"))
                narration_features.append(feature)
            except:
                print(f"Bad narration: {nid}")

        # Load metadata
        narrations = self.annotations[self.annotations['unique_narration_id'].isin(narration_ids)]
        narration_texts, tokens, starts, ends  = [], [], [], []
        for _, row in narrations.iterrows():
            #token = self.tokenizer(row['narration'], max_length=32, truncation=True)['input_ids']
            sec_start = max((int(row['start_frame']) / self.fps) - start_sec, 0)
            sec_end = min((int(row['end_frame']) / self.fps) - start_sec, self.duration)
            #if isinstance(self.tokenizer, Word2VecTokenizer) and (sum(token) == 0):  # all words are stop words
            #        break
            narration_texts.append(row['narration'])
            #tokens.append(torch.tensor(token))
            starts.append(sec_start)
            ends.append(sec_end)


        #truncate the narrations if too many:
        narration_features = narration_features[:self.duration]
        narration_texts = narration_texts[:self.duration]
        #unnorm_starts = unnorm_starts[:self.duration]
        #unnorm_ends = unnorm_ends[:self.duration]
        starts = starts[:self.duration]
        ends = ends[:self.duration]

        padded_narration_features = torch.zeros(int(self.duration), 1, 4096) 
        narration_padding_mask = torch.ones(int(self.duration), dtype=torch.bool)

        if narration_features:
            padded_narration_features[:len(narration_features),::] = torch.stack(narration_features)
            narration_padding_mask[:len(narration_features)] = 0
        
        metadata = {"narrations": narration_texts, 
                    "video_id": video_id, 
                    "exo_camera": exo_cams[0], 
                    "start_sec": start_sec,
                    #"per_second_views": per_second_views
                    }

        output_dict = {
            'video': video_features.squeeze(1),
            'padding_mask': torch.zeros(video_features.size(0)).long(),
            'start': starts,
            'end': ends,
            'narration_features': padded_narration_features.squeeze(1),
            'narration_padding_mask': narration_padding_mask,
            'metadata' : metadata
        }
        return output_dict