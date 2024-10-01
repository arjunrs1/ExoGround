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

class EgoExo4DDataLoader(Dataset):
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
                stitched_best_exo_distill=False,
                exo_mode="all",
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
        self.stitched_best_exo_distill = stitched_best_exo_distill
        self.exo_mode = exo_mode
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

        assert not ((self.views == "ego") and (self.use_distill_nce_loss)) #We cannot train on ego only and distill from ego view simultaneously

        if self.curriculum_train:
            assert self.exo_mode == "all" # curriculum training only on all views (not best, worst, random)
            assert self.split == "train"
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
        if self.multi_view or self.multi_view_single_exo_inference:
            if self.multi_view_egoexo:
                self.view_map = {"aria": 0, "cam01": 1, "gp01": 1, "cam02": 2, "gp02":2, "cam03": 3, "gp03": 3, "cam04": 4, "gp04": 4, "cam05": 5, "gp05": 5, "gp06": 6}
            else:
                self.view_map = {"cam01": 0, "gp01": 0, "cam02": 1, "gp02":1, "cam03": 2, "gp03": 2, "cam04": 3, "gp04": 3, "cam05": 4, "gp05": 4, "gp06": 5}
        self.current_phase = 0
        self.window_csv_path = os.path.join(self.base_path, f'{split}_{views}_ks={use_keysteps}_ct={curriculum_train}_exos={exo_mode}_windows.csv')
        self.windows_cam_distances_path = os.path.join(self.base_path, f'{split}_{views}_ks={use_keysteps}_ct={curriculum_train}_cam_dists.csv')
        self.precompute_windows()
        if self.curriculum_train:
            assert ((self.split == "train") and (self.views == "all"))
            self.cam_distances = pd.read_csv(self.windows_cam_distances_path)
            self.windows['cam_ego_distance'] = self.cam_distances['cam_ego_distance']
            self.windows.sort_values(by='cam_ego_distance', inplace=True)
            self.windows.to_csv("/private/home/arjunrs1/egoexo4d_features/sorted.csv", index=False)
            self.windows.drop(columns=['cam_ego_distance'], inplace=True)

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

    def set_phase(self, phase):
        self.current_phase = phase

    def camera_view_order(self, take_uid, cam_list, start_sec, end_sec, full_ego_cam_name, ego_cam_ray_point=0.7):
        try:
            take_cam_path = f"/datasets01/egoexo4d/v2/annotations/ego_pose/{self.take_uid_cam_pose_split_map[take_uid]}/camera_pose"
            with open(os.path.join(take_cam_path, f"{take_uid}.json"), "rb") as f:
                camera_data = json.load(f)
        except:
            print(f"Could not find camera parameters for {take_uid}")
            assert full_ego_cam_name in cam_list
            cam_list.remove(full_ego_cam_name)
            cam_list.insert(0, full_ego_cam_name)
            sorted_dict = {c: cam_list.index(c) for c in cam_list}
            return cam_list[::-1], sorted_dict

        cam_positions = []
        all_cam_labels = []
        rotations = []
        frame_idx = int((start_sec + ((end_sec - start_sec) / 2)) * self.fps) #use average frame index in window
        ego_cam = None
        for cam, details in camera_data.items():
            try:
                if cam.lower().startswith("aria"):
                    extrinsic = np.array(details['camera_extrinsics'][f"{frame_idx}"])
                    ego_cam = cam
                elif (cam.lower().startswith("cam") or cam.lower().startswith("gp")):
                    extrinsic = np.array(details['camera_extrinsics'])
                else:
                    #metadata key should be ignored
                    continue
            except:
                print(f"Could not get parameters for {cam} with take uid {take_uid}")
                continue
            extrinsic = np.linalg.inv(np.vstack([extrinsic, [0, 0, 0, 1]]))[:3,:]
            translation = extrinsic[:, -1]
            rotation = extrinsic[:, :3]

            cam_positions.append(translation)
            all_cam_labels.append(cam)
            rotations.append(rotation)

        cam_positions = np.array(cam_positions)
        rotations = np.array(rotations)
        ego_index = all_cam_labels.index(ego_cam)

        # Compute the vector between that point and all the exocentric camera positions
        point_X_meters = cam_positions[ego_index] + ego_cam_ray_point * np.dot(rotations[ego_index], [0,0,1])
        vectors_to_exo_cams = point_X_meters - cam_positions
        orientation_vectors = np.dot(rotations, [0,0,1])
        # Compute the cosine similarity between each exocentric camera's orientation vector and its vector computed in step (2)
        cosine_similarities = np.array(F.cosine_similarity(torch.tensor(orientation_vectors), torch.tensor(vectors_to_exo_cams)))
        # Compute the x-y cosine similarity between each exo camera's viewing orientation and the ego camera's viewing orientation to determine ones in front or behind
        x_y_cosine_similarities = np.dot(orientation_vectors[:, :2], orientation_vectors[ego_index, :2]) / (
            np.linalg.norm(orientation_vectors[:, :2], axis=1) * np.linalg.norm(orientation_vectors[ego_index, :2]))
        # Split the x_y_cosine similarities into negative and positive groups
        negative_group = np.where(x_y_cosine_similarities > 0)[0]
        positive_group = np.where(x_y_cosine_similarities <= 0)[0]
        # Use the cosine_similarity array to further sort within each group
        negative_group_final = np.argsort(cosine_similarities[negative_group])[::-1]
        positive_group_final = np.argsort(cosine_similarities[positive_group])[::-1]
        # Combine the two sorted lists
        combined_list = np.concatenate((positive_group[positive_group_final], negative_group[negative_group_final]))
        
        sorted_cams = list(np.take(all_cam_labels, combined_list))
        sorted_cams.remove(ego_cam)
        sorted_cams.insert(0,full_ego_cam_name)
        cam_distances = {c: sorted_cams.index(c) for c in sorted_cams}
        sorted_cams = sorted_cams[::-1]
        return sorted_cams, cam_distances

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
                            if self.split == "train":
                                sorted_cams, cam_distances = self.camera_view_order(take_uid, cams, start_sec, end_sec, ego_cam)
                                far_close_pairs = list(itertools.combinations(sorted_cams, 2))
                                if self.stitched_best_exo_distill:
                                    for cam1 in sorted_cams:
                                        windows.append([video_id, cam1, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                                        cam_dists.append(cam_distances[cam1])
                                else:
                                    for cam1, cam2 in far_close_pairs:
                                        windows.append([video_id, cam1, cam2, start_sec, end_sec, ','.join(narration_ids)])
                                        cam_dists.append(cam_distances[cam1])
                                if ego_cam in cams:
                                    windows.append([video_id, ego_cam, ego_cam, start_sec, end_sec, ','.join(narration_ids)])
                                    cam_dists.append(0)
                            else: 
                                #for val/test set, we should only include each view once (and only exos)
                                for camera in exo_cams:
                                    windows.append([video_id, camera, camera, start_sec, end_sec, ','.join(narration_ids)])
                                     
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

    def get_view_idx(self, exo_cam):
        if "aria" in exo_cam.lower():
            assert self.multi_view_egoexo
            return 0
        return self.view_map[exo_cam]

    def create_video_mask(self, exo_cams):
        num_views = len(exo_cams)

        #Initialize mask with all available views:
        mask = torch.ones(self.num_max_views * self.duration, dtype=torch.bool)
        for cam in exo_cams:
            start_idx = self.get_view_idx(cam) * self.duration
            end_idx = start_idx + self.duration
            mask[start_idx:end_idx] = False

        #if masking, then mask out from existing views:
        if self.current_phase > 0:
            if num_views == self.num_max_views:
                masked_views = np.random.choice(exo_cams, self.current_phase, replace=False)
            elif num_views < self.num_max_views and self.current_phase > (self.num_max_views-num_views):
                masked_views = np.random.choice(exo_cams, self.current_phase-(self.num_max_views-num_views), replace=False)
            else:
                masked_views = None
            if masked_views is not None:
                for view in masked_views:
                    start_idx = self.get_view_idx(view) * self.duration
                    end_idx = start_idx + self.duration
                    mask[start_idx:end_idx] = True
        return mask

    def create_view_available_mask(self, exo_cams):
        mask = torch.zeros(self.num_max_views * self.duration, dtype=torch.bool)
        for cam in exo_cams:
            start_idx = self.get_view_idx(cam) * self.duration
            end_idx = start_idx + self.duration
            mask[start_idx:end_idx] = True
        return mask

    def get_distill_video_features(self, video_id, ego_cam, take_ego_id, start_sec, end_sec):
        #TODO: Speed this up -> Cache the rankings?
        distill_feature_views = [ego_cam for _ in range(self.duration)]
        # Load default video features for the specified duration
        default_features_path = os.path.join(self.video_feature_path, f"{take_ego_id}.pt")
        try:
            default_features = torch.load(default_features_path)[start_sec:end_sec]
        except Exception as e:
            print(f"Error loading default features from {default_features_path}: {e}")
            return None
        stitched_features = default_features.clone()
        take_uid = self.split_data[self.split_data['take_name'] == video_id]['take_uid'].iloc[0]
        atomic_narrations = self.atomic_descriptions_train.get(str(take_uid), [])
        if atomic_narrations:
            descriptions = atomic_narrations[0].get('descriptions', [])
            for narr in descriptions:
                try:
                    timestamp = int(round(narr['timestamp']))
                    if start_sec <= timestamp < end_sec:
                        feat_idx = timestamp - start_sec
                        best_exo = narr['best_exo']['cam_id']
                        take_exo_id = f"{video_id}_{best_exo}"
                        exo_features_path = os.path.join(self.video_feature_path, f"{take_exo_id}.pt")
                        exo_features = torch.load(exo_features_path)
                        stitched_features[feat_idx] = exo_features[timestamp]
                        distill_feature_views[feat_idx] = best_exo
                except Exception as e:
                    pass
        return stitched_features

    def __getitem__(self, idx):
        window = self.windows.iloc[idx]
        video_id, exo_cams, ego_cam, start_sec, end_sec = window['video_id'], window['exo_cam'], window['ego_cam'], window['start_sec'], window['end_sec']
        take_ego_id = f"{video_id}_{ego_cam}"
        narration_ids = window['narration_ids'].split(',')
        exo_cams = ast.literal_eval(exo_cams) if self.multi_view else [exo_cams]
        
        # Load video features
        video_features_list = []
        for exo_cam in exo_cams:
            take_exo_id = f"{video_id}_{exo_cam}"
            features = torch.load(os.path.join(self.video_feature_path, f"{take_exo_id}.pt"))[start_sec:end_sec]
            video_features_list.append(features)
    
        if self.multi_view:
            full_video_feat_len = (self.num_max_views)*self.duration
            video_features = torch.ones(full_video_feat_len, features.shape[-1])
            for cam, feats in zip(exo_cams, video_features_list):
                start_idx = self.get_view_idx(cam) * self.duration
                end_idx = start_idx + self.duration
            video_features[start_idx:end_idx,:] = feats
        else:
            video_features = torch.cat(video_features_list, dim=0)

        #Pad single exo view according to multi-view setting for evaluating multi-view model on single-view inference mode
        if self.multi_view_single_exo_inference:
            assert len(exo_cams) == 1
            index = self.get_view_idx(exo_cams[0])
            rows_left = index * self.duration
            rows_right = (self.num_max_views-index-1) * self.duration
            l_pad = torch.ones(rows_left, video_features.shape[-1])
            r_pad = torch.ones(rows_right, video_features.shape[-1])
            video_features = torch.cat((l_pad, video_features, r_pad), dim=0)
            single_exo_mask = torch.ones(self.num_max_views * self.duration, dtype=torch.bool)
            exo_view_end_idx = rows_left + self.duration
            single_exo_mask[rows_left:exo_view_end_idx] = False
            #Mask is false where valid video features are, and True otherwise (padding mask format)
            

        if self.use_distill_nce_loss:
            if self.stitched_best_exo_distill:
                ego_video_features = self.get_distill_video_features(video_id, ego_cam, take_ego_id, start_sec, end_sec)
            else:
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

        # Randomize the order of narrations if the flag is set
        if self.randomize_narration_order:
            combined = list(zip(narration_texts, starts, ends, narration_features))
            random.shuffle(combined)
            narration_texts, starts, ends, narration_features = zip(*combined)
            narration_texts, starts, ends, narration_features = list(narration_texts), list(starts), list(ends), list(narration_features)

        padded_narration_features = torch.zeros(int(self.duration), 1, 4096) 
        padded_starts = torch.zeros(int(self.duration),1)
        padded_ends = torch.zeros(int(self.duration),1)
        narration_padding_mask = torch.ones(int(self.duration), dtype=torch.bool)

        if narration_features:
            padded_starts[:len(narration_features)] = torch.tensor(starts).unsqueeze(1)
            padded_ends[:len(narration_features)] = torch.tensor(ends).unsqueeze(1)
            padded_narration_features[:len(narration_features),::] = torch.stack(narration_features)
            narration_padding_mask[:len(narration_features)] = 0
        
        metadata = {"narrations": narration_texts, "video_id": video_id, "exo_camera": exo_cams[0], "start_sec": start_sec}
        output_dict = {
            'video_features': video_features.squeeze(1),
            'video_padding_mask': self.create_video_mask(exo_cams) if self.multi_view else
             (single_exo_mask if self.multi_view_single_exo_inference else torch.zeros(video_features.size(0), dtype=torch.bool)),
            'narration_features': padded_narration_features.squeeze(1),
            'narration_padding_mask': narration_padding_mask,
            'starts': padded_starts.squeeze(1),
            'ends': padded_ends.squeeze(1),
            'metadata' : metadata
        }

        if self.multi_view:
                output_dict['view_available_mask'] = self.create_view_available_mask(exo_cams)
        elif self.multi_view_single_exo_inference:
                output_dict['view_available_mask'] = ~output_dict['video_padding_mask']

        if self.use_audio:
            output_dict['audio_padding_mask'] = torch.zeros(audio_features.size(0), dtype=torch.bool)
            output_dict['audio_features'] = audio_features.squeeze(1)
        
        if self.use_distill_nce_loss:
            output_dict['ego_video_features'] = ego_video_features.squeeze(1)

        if self.use_center_duration:
            output_dict['mean'] = (output_dict['starts'] + output_dict['ends']) / 2
            output_dict['duration'] = torch.abs(output_dict['ends']-output_dict['starts'])

        return output_dict