from mlm.src.datasets import ASLWikiDataset
import re
from collections import Counter, defaultdict
import json
import numpy as np
import pandas as pd


def tokenize_sentence(sentence):
    #TODO: Lemmatize words
    
    # Remove punctuation using regular expressions
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Convert to lowercase
    sentence = sentence.lower()
    # Tokenize the sentence into words
    words = sentence.split()
    return words

def word_stats(ds, filename):
    word_counts = defaultdict(lambda: defaultdict(Counter))

    for data in ds:
        words = tokenize_sentence(data.sentence)
        for word in words:
            word_counts[word]['total']['total'] += 1
            word_counts[word]['user'][data.user] += 1
            word_counts[word]['article'][data.article_name] += 1
            word_counts[word]['video'][data.filename] += 1
    
    with open(filename, 'w') as f:
        json.dump(word_counts, f, indent=4)

def get_consecutive_subarrays(arr):
    result = []
    subarray = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1] + 1:
            subarray.append(arr[i])
        else:
            result.append(subarray)
            subarray = [arr[i]]
    
    result.append(subarray)  # Append the last subarray
    return result

def add_zero_stats(zero_stats, part_keypoints, data, body_part):
    zero_segments = np.where(np.all(part_keypoints == 0, axis=(1, 2)))[0]
    if len(zero_segments) == 0:
        return zero_stats
    zero_segments = get_consecutive_subarrays(zero_segments)
    
    for segment in zero_segments:
        if segment[0] == 0:
            zero_stats['segment type'].append('start')
        elif segment[-1] == len(part_keypoints) - 1:
            zero_stats['segment type'].append('end')
        else:
            zero_stats['segment type'].append('middle')
        zero_stats['segment length'].append(len(segment))
        zero_stats['body part'].append(body_part)
        zero_stats['segment span'].append((segment[0], segment[-1]))
        zero_stats['video length'].append(len(part_keypoints))
        zero_stats['user'].append(data.user)
        zero_stats['article'].append(data.article_name)
        zero_stats['video'].append(data.filename)
    return zero_stats

def missing_keypoints_stats(ds, filename):
    missing_stats = defaultdict(list)
    for sample in ds:
        keypoints = sample.keypoints
        posedata = keypoints[:, 0:33, :]
        rhdata = keypoints[:, 33:54, :]
        lhdata = keypoints[:, 54:75, :]
        facedata = keypoints[:, 75:, :]

        missing_stats = add_zero_stats(missing_stats, posedata, sample, 'pose')
        missing_stats = add_zero_stats(missing_stats, rhdata, sample, 'rh')
        missing_stats = add_zero_stats(missing_stats, lhdata, sample, 'lh')
        missing_stats = add_zero_stats(missing_stats, facedata, sample, 'face')

        # with open(f'{filename}.json', 'w') as f:
        #     json.dump(missing_stats, f, indent=4)
        
        df = pd.DataFrame(missing_stats)
        df.to_csv(f'{filename}.csv', index=False)

if __name__ == "__main__":
    datadir = '/home/lualex/PaidWikiDataset/videos/'
    label_file = '/home/lualex/PaidWikiDataset/videos.csv'

    ds = ASLWikiDataset(datadir=datadir, label_file=label_file, window_stride=10, return_videos=False, return_windows=False, mini_dataset=False)

    # word_stats(ds, 'word_stats.json')
    missing_keypoints_stats(ds, 'missing_frames_stats')