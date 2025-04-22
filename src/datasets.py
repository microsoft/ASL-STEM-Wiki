import os
import csv
import re
import random
import math
import ast
from dataclasses import dataclass
# import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utl
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoTokenizer

DEBUG = False

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

@dataclass
class ASLVideo:
    user: str
    article_name: str
    section_index: int
    sentence_index: int
    filename: str
    sentence: str
    video: torch.Tensor
    keypoints: torch.Tensor
    fs_span: list = None
    fs_text: list = None

class ASLWikiDataset(data_utl.Dataset):
    def __init__(self, datadir, label_file, transforms=None, return_videos=False, return_windows=False, window_size=64, window_stride=32, 
                 min_frames=0, drop_face=True, drop_legs=True, eval_article=None, valid=False):
        self.transforms = transforms
        self.datadir = datadir

        if eval_article is None:
            data = np.array([x for x in csv.reader(open(label_file))])[1:]
        else:
            if valid: #only get eval article
                data = np.array([x for x in csv.reader(open(label_file)) if x[1] == eval_article.replace('_', ' ')])[1:]
            else: #get all others
                data = np.array([x for x in csv.reader(open(label_file)) if x[1] != eval_article.replace('_', ' ')])[1:]

        try:
            self.video_df = pd.DataFrame(data, columns=["user",
                                                        "articleName",
                                                        "sectionIndex",
                                                        "sentenceIndex",
                                                        "filename",
                                                        "sentence",
                                                        "fs_span",
                                                        "fs_text"])
        except: 
            self.video_df = pd.DataFrame(data, columns=["user",
                                                        "articleName",
                                                        "sectionIndex",
                                                        "sentenceIndex",
                                                        "filename",
                                                        "videoLength",
                                                        "sentence"])
        # filter out videos with less than min_frames
        # if not DEBUG:
        #     print(f"Filtering videos with less than {min_frames} frames")
        #     if min_frames > 0:
        #         self.video_df['num_frames'] = self.video_df['filename'].apply(lambda x: len(np.load(self.datadir.replace('/videos/', '/keypoints/') + x.replace('/videos/', '/keypoints/').replace('.mp4', '.npy'))))
        #         self.video_df = self.video_df[self.video_df['num_frames'] >= min_frames]
        #     print(f"Filtered {len(data) - len(self.video_df)} videos")

        self.return_windows = return_windows
        self.window_size = window_size
        self.window_stride = window_stride
        self.return_videos = return_videos
        self.min_frames = min_frames
        self.drop_face = drop_face
        self.drop_legs = drop_legs


    def __getitem__(self, index):
        curr_video = self.video_df.iloc[index]
        video_path = self.datadir + curr_video['filename']
        
        keypoints_path = video_path.replace('/videos/', '/keypoints/').replace('.mp4', '.npy')
        if DEBUG:
            # num_frames = 30 # np.random.randint(100, 400)
            ret_keypoints = np.random.rand(max(2000, self.min_frames + 10), 75, 2)
            # random_file = random.choice(os.listdir('/scratch/users/kayoyin/asl-wiki/google-fs/supplemental_keypoints'))
            # file_path = os.path.join('/scratch/users/kayoyin/asl-wiki/google-fs/supplemental_keypoints', random_file)

            # ret_keypoints = np.load(file_path)[:,:,:2]
            # left_hand = ret_keypoints[:, 33:54, :]
            # right_hand = ret_keypoints[:, 54:75, :]
            # ret_keypoints[:, 33:54, :] = right_hand
            # ret_keypoints[:, 54:75, :] = left_hand
            # ret_keypoints = ret_keypoints[:30] # shorten to 30 frames
            
        else:
            ret_keypoints = np.load(keypoints_path)
        
        if len(ret_keypoints) < self.min_frames:
            print(f"input {index}, {video_path}, {curr_video['articleName']}, {ret_keypoints.shape}, min_frames {self.min_frames}")
            # print(f"Skipping video {video_path} with {len(ret_keypoints)} frames, less than {self.min_frames}")
            return self.__getitem__(index+1)
        start, end = 0, len(ret_keypoints)
        if self.drop_face:
            ret_keypoints = ret_keypoints[:, :75, :]
        if self.drop_legs:
            #TODO
            pass

        ret_keypoints = np.nan_to_num(ret_keypoints, nan=0)

        if self.return_videos:
            video = load_rgb_frames_from_video(video_path)[start:end]

            if self.return_windows:
                windows = []
                for start_frame in range (0, len(video), self.window_stride):
                    curr_frames = video[start_frame : start_frame + self.window_size]
                    # captures what happens with the last set of frames - wheel this back to get 64 frames
                    if len(curr_frames) != self.window_size:
                        curr_frames = video[len(video) - self.window_size:]
                        # captures the case if the video itself is less than 64 frames
                        if len(curr_frames) != self.window_size:
                            curr_frames = self.pad(video[start_frame : start_frame + self.window_size],
                                                self.window_size)
                    if self.transforms:
                        curr_frames = self.transforms(curr_frames)
                    curr_frames = video_to_tensor(curr_frames)
                    windows.append(curr_frames)
                ret_video = torch.stack(windows)
            else:
                if self.transforms:
                    video = self.transforms(video)
                ret_video = video_to_tensor(video)
        else:
            ret_video = None

        try:
            return ASLVideo(user=curr_video['user'], 
                            article_name=curr_video['articleName'],
                            section_index=curr_video['sectionIndex'], 
                            sentence_index=curr_video['sentenceIndex'],
                            filename=curr_video['filename'], 
                            sentence=curr_video['sentence'], 
                            video=ret_video,
                            keypoints=ret_keypoints,
                            fs_span=ast.literal_eval(curr_video['fs_span']),
                            fs_text=ast.literal_eval(curr_video['fs_text']))
        except:
            return ASLVideo(user=curr_video['user'], 
                            article_name=curr_video['articleName'],
                            section_index=curr_video['sectionIndex'], 
                            sentence_index=curr_video['sentenceIndex'],
                            filename=curr_video['filename'], 
                            sentence=curr_video['sentence'], 
                            video=ret_video,
                            keypoints=ret_keypoints)
                            
    def __len__(self):
        return len(self.video_df)

    def pad(self, imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]
            if num_padding:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs
        return padded_imgs
    
class HeuristicDetectDataset(Dataset):
    """
    Dataset for heuristic model.
    Returns video, sentence, detection labels, alignment labels.
    """
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        frames = torch.Tensor(data.keypoints)
        
        sentence = data.sentence.lower()

        detect_label = torch.zeros(len(frames))
        for span in data.fs_span:
            detect_label[span[0]:span[1]] = 1
        
        # phrase = str(data.fs_text[0]).lower().strip()
        # start_index = sentence.find(phrase)
        # end_index = start_index + len(phrase)
        
        # align_label = torch.zeros(len(sentence))
        # align_label[start_index:end_index] = 1

        return frames, sentence, detect_label.to(torch.long)#, align_label.to(torch.long)

class SpanTextDataset(Dataset):
    """
    Dataset for task where the model, given a fingerspelling clip and a sentence,
    has to predict the text span of the sentence the clip corresponds to.
    """
    def __init__(self, ds, clip_length, seq_length):
        self.ds = ds
        self.clip_length = clip_length
        self.seq_length = seq_length
        self.tokenizer = AutoTokenizer.from_pretrained("google/canine-c")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        frames = torch.Tensor(data.keypoints)
        frame_start, frame_end = data.fs_span[0]
        
        frames = frames[frame_start:frame_end]
        if len(frames) == 0:
            print("zero frame start, frame end", frame_start, frame_end, len(data.keypoints))
        if len(frames) < self.clip_length:
            try:
                frames = self.pad_frames(frames)
            except:
                return self.__getitem__(idx-1) #TODO: look into why we have clip with 0 frames
        elif len(frames) > self.clip_length:
            frames = self.truncate_frames(frames)

        sentence = data.sentence.lower()
        phrase = str(data.fs_text[0]).lower().strip()
        start_index = sentence.find(phrase)
        if start_index == -1:
            print('ERROR start index:', sentence, phrase)
            start_index = 0
        end_index = start_index + len(phrase)
        if sentence[start_index:end_index] != phrase:
            pass
            # print('ERROR word:', sentence, phrase, sentence[start_index:end_index])
        end_index = min(end_index, self.seq_length)

        sentence = self.tokenizer.encode(sentence)
        sentence_len = len(sentence)
        sentence = sentence[:self.seq_length] + [self.tokenizer.pad_token_id] * (self.seq_length - len(sentence)) if len(sentence) < self.seq_length else sentence[:self.seq_length]
        sentence = torch.tensor(sentence, dtype=torch.long)
        attention_mask = torch.ones_like(sentence)
        attention_mask[sentence == self.tokenizer.pad_token_id] = 0

        label = torch.zeros(self.seq_length)
        label[start_index:end_index] = 1
        label[sentence_len:] = 2

        return frames, sentence, attention_mask, label.to(torch.long), start_index, end_index

    def pad_frames(self, frames):
        num_frames = len(frames)
        padding = torch.stack([torch.zeros_like(frames[0]) for _ in range(self.clip_length - num_frames)])
        padded_frames = torch.cat([frames, padding])
        return padded_frames
    
    def truncate_frames(self, frames):
        truncated_frames = frames[:self.clip_length]
        return truncated_frames

class SpanVideoDataset(Dataset):
    """
    Dataset for task where the model, given an ASL clip,
    has to predict the video span containing fingerspelling.
    """
    def __init__(self, ds, clip_length, seq_length):
        self.ds = ds
        self.clip_length = clip_length
        self.seq_length = seq_length
        self.tokenizer = AutoTokenizer.from_pretrained("google/canine-c")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        frames = torch.Tensor(data.keypoints)
        if len(frames) < self.clip_length:
            try:
                frames = self.pad_frames(frames)
            except:
                return self.__getitem__(idx-1) #TODO: look into why we have clip with 0 frames
        elif len(frames) > self.clip_length:
            frames = self.truncate_frames(frames)

        
        sentence = self.tokenizer.encode(data.sentence.lower())
        sentence = sentence[:self.seq_length] + [0] * (self.seq_length - len(sentence)) if len(sentence) < self.seq_length else sentence[:self.seq_length]
        sentence = torch.tensor(sentence, dtype=torch.long)
        attention_mask = torch.ones_like(sentence)
        attention_mask[sentence == 0] = 0

        label = torch.zeros(self.clip_length)
        for span in data.fs_span:
            label[span[0]:span[1]] = 1
        label[len(frames):] = 2
        return frames, sentence, attention_mask, label.to(torch.long)

    def pad_frames(self, frames):
        num_frames = len(frames)
        padding = torch.stack([torch.zeros_like(frames[0]) for _ in range(self.clip_length - num_frames)])
        padded_frames = torch.cat([frames, padding])
        return padded_frames
    
    def truncate_frames(self, frames):
        truncated_frames = frames[:self.clip_length]
        return truncated_frames
    
class ContraTextDataset(Dataset):
    """
    Dataset for task where the model, given a fingerspelling clip,
    has to predict whether the clip corresponds to the first or the
    second English phrase.
    """
    def __init__(self, ds, clip_length, seq_length):
        self.ds = ds
        self.clip_length = clip_length
        self.seq_length = seq_length
        self.phrases = ds.video_df.sentenceIndex.unique()
        self.tokenizer = AutoTokenizer.from_pretrained("google/canine-c")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        frames = torch.Tensor(data.keypoints)
        if len(frames) < self.clip_length:
            try:
                frames = self.pad_frames(frames)
            except:
                return self.__getitem__(idx-1) #TODO: look into why we have clip with 0 frames
        elif len(frames) > self.clip_length:
            frames = self.truncate_frames(frames)

        text1 = data.sentence_index.lower()
        text2 = random.choice(self.phrases).lower()
        while text1 == text2:
            text2 = random.choice(self.phrases).lower()

        text1 = self.tokenizer.encode(text1)
        text1 = text1[:self.seq_length] + [0] * (self.seq_length - len(text1)) if len(text1) < self.seq_length else text1[:self.seq_length]
        text1 = torch.tensor(text1, dtype=torch.long)
        text2 = self.tokenizer.encode(text2)
        text2 = text2[:self.seq_length] + [0] * (self.seq_length - len(text2)) if len(text2) < self.seq_length else text2[:self.seq_length]
        text2 = torch.tensor(text2, dtype=torch.long)
        
        label = torch.randint(low=0, high=2, size=(1,)).item()
        if label == 1:
            text1, text2 = text2, text1
            
        # print(text1.shape, text2.shape)
        text = torch.cat([text1, text2])

        attention_mask = torch.ones_like(text)
        attention_mask[text == 0] = 0

        return frames, text, attention_mask, label

    def pad_frames(self, frames):
        num_frames = len(frames)
        padding = torch.stack([torch.zeros_like(frames[0]) for _ in range(self.clip_length - num_frames)])
        padded_frames = torch.cat([frames, padding])
        return padded_frames
    
    def truncate_frames(self, frames):
        truncated_frames = frames[:self.clip_length]
        return truncated_frames
    
class ContraVideoDataset(Dataset):
    def __init__(self, ds, clip_length, seq_length, offset=50):
        self.ds = ds
        self.clip_length = clip_length
        self.seq_length = seq_length
        self.offset = offset
        self.user_ids = defaultdict(list)
        for i, data in enumerate(self.ds):
            self.user_ids[data.user].append(i)
        self.tokenizer = AutoTokenizer.from_pretrained("google/canine-c")
        # assert self.ds.min_frames > self.clip_length * 2 + self.offset

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        frames = torch.Tensor(data.keypoints)
        label = torch.randint(low=0, high=3, size=(1,)).item() # 0 = different video, 1 = same video first clip, 2 = same video second clip
        
        
        sentence = self.tokenizer.encode(data.sentence.lower())
        sentence = sentence[:self.seq_length] + [0] * (self.seq_length - len(sentence)) if len(sentence) < self.seq_length else sentence[:self.seq_length]
        sentence = torch.tensor(sentence, dtype=torch.long)
        attention_mask = torch.ones_like(sentence)
        attention_mask[sentence == 0] = 0

        if label == 0:
            clip_start = torch.randint(low=0, high=len(frames) - self.clip_length, size=(1,)).item()
            clip = frames[clip_start : clip_start + self.clip_length]
            
            if len(self.user_ids[data.user]) == 1:
                return self.__getitem__(idx)
            
            other_idx = np.random.choice(self.user_ids[data.user])
            while other_idx == idx: # or user != self.ds[other_idx].user: #TODO: maybe not efficient
                other_idx = np.random.choice(self.user_ids[data.user])
            other_frames = torch.Tensor(self.ds[other_idx].keypoints)
            other_clip_start = torch.randint(low=0, high=len(other_frames) - self.clip_length, size=(1,)).item()
            other_clip = other_frames[other_clip_start : other_clip_start + self.clip_length]
        elif label == 1:
            clip_start = torch.randint(low=0, high=len(frames) - self.clip_length * 2 - self.offset, size=(1,)).item()
            clip = frames[clip_start : clip_start + self.clip_length]
            other_clip_start = torch.randint(low=clip_start + self.clip_length + self.offset, high=len(frames) - self.clip_length, size=(1,)).item()
            other_clip = frames[other_clip_start : other_clip_start + self.clip_length]
        else:
            other_clip_start = torch.randint(low=0, high=len(frames) - self.clip_length * 2 - self.offset, size=(1,)).item()
            other_clip = frames[other_clip_start : other_clip_start + self.clip_length]
            clip_start = torch.randint(low=other_clip_start + self.clip_length + self.offset, high=len(frames) - self.clip_length, size=(1,)).item()
            clip = frames[clip_start : clip_start + self.clip_length]

        return clip, other_clip, sentence, attention_mask, label


if __name__ == "__main__":
    from torchvision import transforms
    import videotransforms

    datadir = '/path/to/folder/videos/'
    label_file = '/path/to/csvfile/videos.csv'

    test_transforms = transforms.Compose([videotransforms.BottomCenterCrop(224)])
    ds = ASLWikiDataset(datadir=datadir, transforms=test_transforms, label_file=label_file, window_stride=10)

    for i in range (0, 100):
        vid, name, sentence = ds.__getitem__(i)
        print (vid.size())
