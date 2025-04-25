from datasets import *
from metrics import *

from collections import namedtuple
import json
import re
import torch

def compute_iou(prediction, label):
    assert len(prediction) == len(label), "Sequences must be of equal length"
    
    intersection = sum(p == 1 and l == 1 for p, l in zip(prediction, label))
    union = sum(p == 1 or l == 1 for p, l in zip(prediction, label))
    
    iou_score = intersection / union if union != 0 else 0
    return iou_score

def count_groups_of_1s(arr):
    arr = list(arr)
    count = 0
    in_group = False

    for num in arr:
        if num == 1 and not in_group:
            count += 1
            in_group = True
        elif num == 0:
            in_group = False

    return count

def fs_aligner(sentence, label, wiki_words):
    num_words = count_groups_of_1s(label)

    words = list(map(lambda x: re.sub(r'[^a-zA-Z ]', '', x).lower(), sentence.split()))
    word_freqs = dict()
    for word in words:
        word_freqs[word] = wiki_words.get(word, 0)
    low_freq_words = sorted(word_freqs, key=word_freqs.get)[:num_words]

    align_preds = torch.zeros(len(label))
    for phrase in low_freq_words:
        phrase = str(phrase).lower().strip()
        start_index = sentence.find(phrase)
        end_index = start_index + len(phrase)
        align_preds[start_index:end_index] = 1

    return align_preds

def run_heuristic_align(cfg):
    with open('wikipedia_words.json', 'r') as f:
        wiki_words = json.load(f)

    detect_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0)
    detect_dataset = HeuristicDataset(detect_ds)

    align_preds = []
    align_labels = []
    align_iou = []

    cur_sent = ''

    sent_count = 0

    for i in range(len(detect_dataset)):
        frames, sentence, detect_label, align_label = detect_dataset[i]
        if sentence == cur_sent:
            sent_count += 1
        else:
            sent_count = 0
            cur_sent = sentence
        preds = fs_aligner(sentence, align_label, wiki_words)

        align_preds.append(list(map(lambda x: str(x.item()), preds)))
        align_labels.append(list(map(lambda x: str(x.item()), align_label)))

        score = compute_iou(preds, align_label)
        align_iou.append(score)

    # Save align_preds
    with open('align_preds.txt', 'w') as f:
        for pred in align_preds:
            f.write(' '.join(pred) + '\n')
    
    return sum(align_iou) / len(align_iou)

if __name__ == '__main__':
    cfg = namedtuple('Config', ['seed', 'meta', 'data', 'model', 'train', 'wandb', 'finetune'])
    cfg.seed = 42
    cfg.data = namedtuple('Data', ['ft_datadir', 'ft_label_file'])
    cfg.data.ft_datadir = 'ASL_STEM_Wiki_data/videos/'
    cfg.data.ft_label_file = 'fs-annotations/train.csv'

    iou = run_heuristic_align(cfg)
    print(iou)
    