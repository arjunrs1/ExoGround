import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
import json

class EgoExo4DDataLoader(Dataset):
    def __init__(self,
                split='train',
                duration=20,
                hop_length=10,
                use_audio=True,
                use_keysteps=False,
                views="exo",
                use_distill_nce_loss=False,
                use_center_duration=False,
                fps=30):
        self.split = split
        self.duration = duration
        self.hop_length = hop_length
        self.use_audio = use_audio
        self.use_keysteps = use_keysteps
        self.views = views
        self.use_distill_nce_loss = use_distill_nce_loss
        self.use_center_duration = use_center_duration
        self.fps = fps
        self.base_path = '/private/home/arjunrs1/egoexo4d_features'
        self.vid_feat_rel_path = "checkpoint/yalesong/EgoExo4D_EgoVLPv2_arxivC/extracted_features_egovlp2/EgoVLPv2_Pretraining_bs512-lr3e_5-Ep20"
        self.split_path = f"/private/home/arjunrs1/exo_narration_grounding/splits/egoexo4d_splits/{split}.csv"
        self.annotation_path = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/narration_annotations/{split}.csv'
        self.keysteps_annotations_path = f'/private/home/arjunrs1/exo_narration_grounding/data_processing/time_interval_annotation_files/keystep_annotations/{split}.csv'

        self.atomic_take_cam_map_train_path = f'/datasets01/egoexo4d/v2/annotations/atomic_descriptions_train.json'
        self.atomic_take_cam_map_val_path = f'/datasets01/egoexo4d/v2/annotations/atomic_descriptions_val.json'

        with open(self.atomic_take_cam_map_train_path, "rb") as f:
            self.atomic_take_cam_map_train = json.load(f)['take_cam_id_map']

        with open(self.atomic_take_cam_map_val_path, "rb") as f:
            self.atomic_take_cam_map_val = json.load(f)['take_cam_id_map']

        assert not ((self.views == "ego") and (self.use_distill_nce_loss)) #We cannot train on ego only and distill from ego view simultaneously

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

        self.window_csv_path = os.path.join(self.base_path, f'{split}_{views}_ks={use_keysteps}_windows.csv')
        self.precompute_windows()

    def precompute_windows(self):
        if not os.path.exists(self.window_csv_path):
            print("Computing windows...")
            windows = []
            for _, row in self.split_data.iterrows():
                video_id = row['take_name']
                take_uid = row['take_uid']
                duration_sec = int(row['duration_sec'])
                exo_cams = list(self.atomic_take_cam_map_train[take_uid].values())
                ego_cam = row['ego_camera_path'].split("/")[-1].split(".")[0]
                if self.views == "exo":
                    cams = exo_cams
                elif self.views == "ego":
                    cams = [ego_cam]
                else:
                    cams = exo_cams
                    cams.append(ego_cam)
                for exo_cam in cams:
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
                            windows.append([video_id, exo_cam, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
            windows_df = pd.DataFrame(windows, columns=['video_id', 'exo_cam', 'ego_cam', 'start_sec', 'end_sec', 'narration_ids'])
            windows_df.to_csv(self.window_csv_path, index=False)
            self.windows = windows_df
        else:
            print("Loading windows...")
            self.windows = pd.read_csv(self.window_csv_path)
        print(f"Number of {self.split} windows:")
        print(len(self.windows))

    def __len__(self):
        return len(self.windows)

    @staticmethod
    def collate_fn(batch):
    # Use the default collate to handle everything except 'metadata'
        batch_rest = [{k: v for k, v in item.items() if k != 'metadata'} for item in batch]
        collated_data = default_collate(batch_rest)
        # Handle the metadata separately
        metadata_keys = batch[0]['metadata'].keys()  # Assuming all items have the same keys
        collated_metadata = {}
        for key in metadata_keys:
            collated_metadata[key] = [item['metadata'][key] for item in batch]
        # Add the collated metadata back to the main data
        collated_data['metadata'] = collated_metadata
        return collated_data

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        video_id, exo_cam, ego_cam, start_sec, end_sec = window['video_id'], window['exo_cam'], window['ego_cam'], window['start_sec'], window['end_sec']
        take_exo_id = f"{video_id}_{exo_cam}"
        take_ego_id = f"{video_id}_{ego_cam}"
        narration_ids = window['narration_ids'].split(',')

        # Load video features
        video_features = torch.load(os.path.join(self.video_feature_path, f"{take_exo_id}.pt"))[start_sec:end_sec]
        if self.use_distill_nce_loss:
            ego_video_features = torch.load(os.path.join(self.video_feature_path, f"{take_ego_id}.pt"))[start_sec:end_sec]
        
        #Load audio features if used
        if self.use_audio:
            full_audio_features = np.load(os.path.join(self.audio_feature_path, f"{take_exo_id}.npy"))
            audio_features = torch.from_numpy(full_audio_features[start_sec:end_sec]).float()
        
        # Load narration features
        narration_features = []
        for nid in narration_ids:
            try:
                feature = torch.load(os.path.join(self.narration_feature_path, video_id, f"{nid}.pt"))
                narration_features.append(feature)
            except:
                print(f"Bad narration: {nid}")

        # Load metadata
        #NOTE: Should we include a narration if its timestamp is in the range, or if it overlaps (start, end) interval is in the range?
        narrations = self.annotations[self.annotations['unique_narration_id'].isin(narration_ids)]
        narration_texts, starts, ends = [], [], []
        for _, row in narrations.iterrows():
            narration_texts.append(row['narration'])
            start = max(((int(row['start_frame']) / self.fps) - start_sec) / self.duration, 0.0)
            end = min(((int(row['end_frame']) / self.fps) - start_sec) / self.duration, 1.0)
            starts.append(start)
            ends.append(end)

        #truncate the narrations if too many:
        narration_features = narration_features[:self.duration]
        narration_texts = narration_texts[:self.duration]
        starts = starts[:self.duration]
        ends = ends[:self.duration]

        padded_narration_features = torch.zeros(int(self.duration), 1, 4096) 
        padded_starts = torch.zeros(int(self.duration),1)
        padded_ends = torch.zeros(int(self.duration),1)
        narration_padding_mask = torch.ones(int(self.duration), dtype=torch.bool)

        if narration_features:
            padded_starts[:len(narration_features)] = torch.tensor(starts).unsqueeze(1)
            padded_ends[:len(narration_features)] = torch.tensor(ends).unsqueeze(1)
            padded_narration_features[:len(narration_features),::] = torch.stack(narration_features)
            narration_padding_mask[:len(narration_features)] = 0
        
        metadata = {"narrations": narration_texts, "video_id": video_id, "exo_camera": exo_cam, "start_sec": start_sec}
        output_dict = {
            'video_features': video_features.squeeze(1),
            "video_padding_mask": torch.zeros(video_features.size(0), dtype=torch.bool),
            'narration_features': padded_narration_features.squeeze(1),
            'narration_padding_mask': narration_padding_mask,
            'starts': padded_starts.squeeze(1),
            'ends': padded_ends.squeeze(1),
            'metadata' : metadata
        }
        if self.use_audio:
            output_dict['audio_padding_mask'] = torch.zeros(audio_features.size(0), dtype=torch.bool)
            output_dict['audio_features'] = audio_features.squeeze(1)
        
        if self.use_distill_nce_loss:
            output_dict['ego_video_features'] = ego_video_features.squeeze(1)

        if self.use_center_duration:
            output_dict['mean'] = (output_dict['starts'] + output_dict['ends']) / 2
            output_dict['duration'] = torch.abs(output_dict['ends']-output_dict['starts'])

        return output_dict