from heuristics import *
from metrics import *
from datasets import *

from pathlib import Path
import numpy as np
import torch
import wandb
from collections import namedtuple

wandb.login(key='cf2b2cf6b147e64676b9725007c2877d0f2d3817', relogin=True)


def scores_detector(scores, threholds, union=True):
    if union:
        rh = (scores['rh_fing_var'] > threholds['rh_fing_var']) or (scores['rh_forearm_var'] < threholds['rh_forearm_var'])
        lh = (scores['lh_fing_var'] > threholds['lh_fing_var']) or (scores['lh_forearm_var'] < threholds['lh_forearm_var'])
    else:
        rh = (scores['rh_fing_var'] > threholds['rh_fing_var']) and (scores['rh_forearm_var'] < threholds['rh_forearm_var'])
        lh = (scores['lh_fing_var'] > threholds['lh_fing_var']) and (scores['lh_forearm_var'] < threholds['lh_forearm_var'])
    if rh:
        return scores['rh_fing_ind'] + scores['rh_fing_var'] + (1 - scores['rh_forearm_var'])
    elif lh:
        return scores['lh_fing_ind'] + scores['lh_fing_var'] + (1 - scores['lh_forearm_var'])
    else:
        return 0
    
def model_detector(frames, model, fs_rep):
    lh_data = frames.clone()
    lh_data[:, 33:54, :] = 0 # mask rh
    lh_data = torch.stack([lh_data,lh_data])
    b, s, k, d = lh_data.shape
    lh_data = lh_data.reshape(b, -1, k).permute([0, 2, 1])
    rep = model(lh_data.to(next(model.parameters()).device), repr=True).detach().cpu().numpy()[0]
    lh_similarity = cosine_similarity(fs_rep, rep)

    rh_data = frames.clone()
    rh_data[:, 54:75, :] = 0 # mask lh
    rh_data = torch.stack([rh_data,rh_data])
    b, s, k, d = rh_data.shape
    rh_data = rh_data.reshape(b, -1, k).permute([0, 2, 1])
    rep = model(rh_data.to(next(model.parameters()).device), repr=True).detach().cpu().numpy()[0]
    rh_similarity = cosine_similarity(fs_rep, rep)

    return max(lh_similarity, rh_similarity), {'rh_similarity': rh_similarity, 'lh_similarity': lh_similarity}


def heuristic_detector(frames, t_fing_i, t_fing_v, t_fore):
    scores = dict()

    hand = np.concatenate((frames[:, 33:54, :],  np.zeros((frames.shape[0], 21, 1))), axis=-1)
    zero_mask = np.all(np.all(hand == 0, axis=1), axis=1)
    hand = hand[~zero_mask]
    if len(hand) == 0:
        scores['rh_fing_ind'] = 0
        scores['rh_fing_var'] = 0
    else:
        hand_angles = to_so3(jnp.array(hand))
        fing_ind = [finger_independence(angles) for angles in hand_angles]
        scores['rh_fing_ind'] = sum(fing_ind) / len(fing_ind)
        scores['rh_fing_var'] = float(np.var(hand_angles, axis=0).mean())

    pose = np.array(frames[:, RIGHT_FOREARM_POINTS, :])
    zero_mask = np.all(np.all(pose == 0, axis=1), axis=1)
    pose = pose[~zero_mask]
    if len(pose) == 0:
        scores['rh_forearm_var'] = 1
    else:
        scores['rh_forearm_var'] = np.var(pose, axis=0).mean()

    hand = np.concatenate((frames[:, 54:75, :],  np.zeros((frames.shape[0], 21, 1))), axis=-1)
    zero_mask = np.all(np.all(hand == 0, axis=1), axis=1)
    hand = hand[~zero_mask]
    if len(hand) == 0:
        scores['lh_fing_ind'] = 0
        scores['lh_fing_var'] = 0
    else:
        hand_angles = to_so3(jnp.array(hand))
        fing_ind = [finger_independence(angles) for angles in hand_angles]
        scores['lh_fing_ind'] = sum(fing_ind) / len(fing_ind)
        scores['lh_fing_var'] = float(np.var(hand_angles, axis=0).mean())

    pose = np.array(frames[:, LEFT_FOREARM_POINTS, :])
    zero_mask = np.all(np.all(pose == 0, axis=1), axis=1)
    pose = pose[~zero_mask]
    if len(pose) == 0:
        scores['lh_forearm_var'] = 1
    else:
        scores['lh_forearm_var'] = np.var(pose, axis=0).mean()

    rh_score = scores['rh_fing_ind'] * t_fing_i + scores['rh_fing_var'] * t_fing_v + (1-scores['rh_forearm_var']) * t_fore
    lh_score = scores['lh_fing_ind'] * t_fing_i + scores['lh_fing_var'] * t_fing_v + (1 - scores['lh_forearm_var']) * t_fore
    score = max(rh_score, lh_score)
    return score, scores

def fs_detector(frames, labels):
    t_fing_i = 1/10
    t_fing_v = 1/10
    t_fore = 4/5
    stride = 10
    clip_length = 20

    thresholds = dict()
    score_values = defaultdict(list)
    num_clusters = 2
    sim_scores = defaultdict(dict)

    start_frame = torch.sum(frames.view(frames.shape[0], -1), dim=1).nonzero()[0].item()
    end_frame = torch.sum(frames.view(frames.shape[0], -1), dim=1).nonzero()[-1].item()

    for cur_frame in range(start_frame, end_frame, stride):
        x = frames[cur_frame:cur_frame+clip_length]
        score, scores = heuristic_detector(x, t_fing_i, t_fing_v, t_fore)
        sim_scores[(cur_frame, cur_frame + clip_length)] = scores

    for span, scores in sim_scores.items():
        for metric, score in scores.items():
            score_values[metric].append(score)

    for metric, scores in score_values.items():
        if len(scores) < 2:
            thresholds[metric] = 0
            continue
        smallest_values_in_clusters  = find_best_clustering(scores, num_clusters)
        thresholds[metric] = max(smallest_values_in_clusters)

    pred_spans = []
    for cur_frame in range(start_frame, end_frame, stride):
        score = scores_detector(sim_scores[(cur_frame, cur_frame+clip_length)], thresholds, union = False)
        if score > 0:
            pred_spans.append([cur_frame, cur_frame + clip_length, score])
    pred_spans = merge_spans(pred_spans)

    pred_labels = torch.zeros_like(labels)
    for s, e, _ in pred_spans:
        pred_labels[s:e] = 1

    return pred_labels

def compute_iou(prediction, label):
    assert len(prediction) == len(label), "Sequences must be of equal length"
    
    intersection = sum(p == 1 and l == 1 for p, l in zip(prediction, label))
    union = sum(p == 1 or l == 1 for p, l in zip(prediction, label))
    
    iou_score = intersection / union if union != 0 else 0
    return iou_score

def run_heuristic_eval(cfg):
    torch.manual_seed(cfg.seed)

    output_path = Path(f'{cfg.train.model_dir}/heuristic-outputs')
    if not output_path.exists():
        output_path.mkdir(parents=True)

    detect_ds = ASLWikiDataset(datadir=cfg.data.ft_datadir, label_file=cfg.data.ft_label_file, min_frames=0)
    detect_dataset = HeuristicDetectDataset(detect_ds)

    name = cfg.wandb.name
    wandb.init(project='aslnn',
               entity='kayo',
               name=name)

    detect_preds = []
    detect_labels = []
    sentences = []
    detect_iou = []


    for i in range(len(detect_dataset)):
        frames, sentence, label = detect_dataset[i]
        preds = fs_detector(frames, label)
        sentences.append(sentence)

        detect_preds.append(list(map(lambda x: str(x.item()), preds)))
        detect_labels.append(list(map(lambda x: str(x.item()), label)))

        score = compute_iou(preds, label)
        wandb.log({'detect-iou': score})
        detect_iou.append(score)

    with open(f'{output_path}/detect_preds.txt', 'w') as f:
        for preds in detect_preds:
            f.write(','.join(preds) + '\n')

    with open(f'{output_path}/detect_labels.txt', 'w') as f:
        for labels in detect_labels:
            f.write(','.join(labels) + '\n')

    with open(f'{output_path}/sentences.txt', 'w') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

    wandb.save(f'{output_path}/detect_preds.txt')
    wandb.save(f'{output_path}/detect_labels.txt')
    wandb.save(f'{output_path}/sentences.txt')

    return sum(detect_iou) / len(detect_iou)


if __name__ == '__main__':
    cfg = namedtuple('Config', ['seed', 'meta', 'data', 'model', 'train', 'wandb', 'finetune'])
    cfg.seed = 42
    cfg.meta = namedtuple('Meta', ['device'])
    cfg.meta.device = 'cuda'
    cfg.data = namedtuple('Data', ['datadir', 'label_file', 'clip_length', 'batch_size', 'ft_datadir', 'ft_label_file'])
    cfg.data.datadir = '/scratch/users/kayoyin/asl-wiki/google-fs/supplemental_videos/'
    cfg.data.label_file = '/scratch/users/kayoyin/asl-wiki/fs-data-remapped/train.csv'
    cfg.data.clip_length = 100
    cfg.data.batch_size = 8
    cfg.data.seq_length = 8
    cfg.data.ft_datadir = '/scratch/users/kayoyin/asl-wiki/google-fs/supplemental_videos'
    cfg.data.ft_label_file = '/scratch/users/kayoyin/asl-wiki/fs-data-remapped/train.csv'
    cfg.data.eval_article = 'EDGE species'
    cfg.model = namedtuple('Model', ['num_keypoints', 'hidden_feature', 'p_dropout', 'num_stages'])
    cfg.model.num_keypoints = 75
    cfg.model.hidden_feature = 768
    cfg.model.p_dropout = 0.1
    cfg.model.num_stages = 2
    cfg.train = namedtuple('Train', ['learning_rate', 'num_epochs', 'model_dir'])
    cfg.train.learning_rate = 1e-3
    cfg.train.num_epochs = 0
    cfg.train.model_dir = 'test'
    cfg.wandb = namedtuple('Wandb', ['name'])
    cfg.wandb.name = 'test'
    cfg.finetune = namedtuple('Finetune', ['num_epochs', 'model_dir'])
    cfg.finetune.num_epochs = 1
    cfg.finetune.model_dir = 'test'

    acc = run_heuristic_eval(cfg)
    print(acc)
    