import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from collections import defaultdict

import json
import glob
import re
import os

import jax
from jax import numpy as jnp
import numpy as np

ONE_HAND_JOINTS = [[ 5,  0, 17],
       [17,  0,  1],
       [ 0,  1,  2],
       [ 1,  2,  3],
       [ 4,  3,  2],
       [ 0,  5,  6],
       [ 6,  5,  9],
       [ 5,  6,  7],
       [ 6,  7,  8],
       [ 5,  9, 10],
       [10,  9, 13],
       [ 9, 10, 11],
       [10, 11, 12],
       [14, 13, 17],
       [17, 13,  9],
       [13, 14, 15],
       [14, 15, 16],
       [18, 17,  0],
       [ 0, 17, 13],
       [17, 18, 19],
       [18, 19, 20]]

# List of list of joints that correspond to same part of each finger
FINGER_JOINTS = [[3,7,11, 15, 19],
                [4,8,12, 16, 20]]

LEFT_FOREARM_JOINTS = [[11, 12, 14],
                       [12, 14, 16]]

RIGHT_FOREARM_JOINTS = [[12, 11, 13],
                        [11, 13, 15]]

LEFT_FOREARM_POINTS = [12, 14, 16] # goes with 54-74

RIGHT_FOREARM_POINTS = [11, 13, 15] # goes with 33-53

HAND_PALM_CONNECTIONS = [(0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17)]
HAND_THUMB_CONNECTIONS = [(1, 2), (2, 3), (3, 4)]
HAND_INDEX_FINGER_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
HAND_MIDDLE_FINGER_CONNECTIONS = [(9, 10), (10, 11), (11, 12)]
HAND_RING_FINGER_CONNECTIONS = [(13, 14), (14, 15), (15, 16)]
HAND_PINKY_FINGER_CONNECTIONS = [(17, 18), (18, 19), (19, 20)]
HAND_CONNECTIONS = HAND_PALM_CONNECTIONS + HAND_THUMB_CONNECTIONS + HAND_INDEX_FINGER_CONNECTIONS + HAND_MIDDLE_FINGER_CONNECTIONS + HAND_RING_FINGER_CONNECTIONS + HAND_PINKY_FINGER_CONNECTIONS

HAND_LENGTH_KEYS = [[ 0,  1],
       [ 0,  5],
       [ 9, 13],
       [13, 17],
       [ 5,  9],
       [ 0, 17],
       [ 1,  2],
       [ 2,  3],
       [ 3,  4],
       [ 5,  6],
       [ 6,  7],
       [ 7,  8],
       [ 9, 10],
       [10, 11],
       [11, 12],
       [13, 14],
       [14, 15],
       [15, 16],
       [17, 18],
       [18, 19],
       [19, 20]]

def se3_to_so3(data):
    """Convert data in SE(3) to SO(3): find R such that R*ba = bc."""
    data = jnp.array(data)
    joints = jnp.array(ONE_HAND_JOINTS)
    
    a = data[joints[0]]
    b = data[joints[1]]
    c = data[joints[2]]

    ba = a - b
    bc = c - b

    ba_mag = jnp.linalg.norm(ba)
    bc_mag = jnp.linalg.norm(bc)

    ba_norm = jax.lax.cond(ba_mag > 0, lambda x: x / ba_mag,
                        lambda x: jnp.zeros_like(x).astype(jnp.float32), ba)
    bc_norm = jax.lax.cond(bc_mag > 0, lambda x: x / bc_mag,
                        lambda x: jnp.zeros_like(x).astype(jnp.float32), bc)

    v = jnp.cross(ba_norm, bc_norm)
    c = jnp.dot(ba_norm, bc_norm)
    vx = jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    def _check_colinear(c):
        return jax.lax.cond(
           c == -1, lambda x: jnp.array([[-1., 0, 0], [0, -1., 0], [0, 0, -1.]]),
           lambda x: jnp.eye(3) + x[0] + jnp.dot(x[0], x[0]) * (1 / (1 + x[1])),
           [vx, c])

    r = jax.lax.cond(
     jnp.all(v == 0), lambda x: jnp.eye(3).astype(jnp.float32),
     _check_colinear, c)
    return jnp.arccos((jnp.trace(r) - 1) / 2)

to_so3 = jax.vmap(jax.vmap(se3_to_so3))


def distance(hand1, hand2):
    if isinstance(hand1, dict):
        hand1 = to_so3(jnp.array(hand1['hand_landmarks']))
    if isinstance(hand2, dict):
        hand2 = to_so3(jnp.array(hand2['hand_landmarks']))
        
#     # Convert angles to vectors on the unit circle
#     vecs1 = np.array([[np.cos(a), np.sin(a)] for a in hand1])
#     vecs2 = np.array([[np.cos(a), np.sin(a)] for a in hand2])

#     # Compute the cosine similarity matrix
#     cos_sim = np.dot(vecs1, vecs2.T) / np.outer(np.linalg.norm(vecs1, axis=1), np.linalg.norm(vecs2, axis=1))

    diffs = np.abs(hand1 - hand2)
    # Handle the case where the difference is greater than pi
    diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)

    return np.mean(diffs, dtype=np.float64)
    
def finger_independence(hand):
    # Higher = more independence
    if isinstance(hand, dict):
        hand = to_so3(jnp.array(hand['hand_landmarks']))
    
    all_score = 0
    for finger_set in FINGER_JOINTS:
        score = []
        for joint_1 in finger_set:
            for joint_2 in finger_set:
                if joint_1 != joint_2:
                    diff = np.abs(hand[joint_1] - hand[joint_2])
                    if diff > np.pi:
                        diff = 2 * np.pi - diff
                    score.append(diff)
        all_score += sum(score) / len(score)

    return all_score / len(FINGER_JOINTS)

def get_scores(lang):
    with open(f'data/{lang}.json','r') as file:
        dataset = json.load(file)

    speaker_scores = dict()
    alphabet = dict()
    for sample in dataset['samples']:
        if sample['letter'] in speaker_scores:
            continue
            
        speaker_scores[sample['letter']] = distance(sample, resting)
        alphabet[sample['letter']] = sample

    finger_ind = dict()
    for char, sample in alphabet.items():
        finger_ind[char] = finger_independence(sample)

    listener_scores = dict()
    for char1, sample1 in alphabet.items():
        for char2, sample2 in alphabet.items():
            if char1 != char2:
                listener_scores[char1+char2 if char1 <= char2 else char2+char1] = distance(sample1, sample2)

    if not os.path.exists(f'scores/{lang}'):
        # Create the directory
        os.makedirs(f'scores/{lang}')

    with open(f'scores/{lang}/speaker_scores.json','w') as file:
        json.dump(speaker_scores, file)
    with open(f'scores/{lang}/finger_ind.json','w') as file:
        json.dump(finger_ind, file)
    with open(f'scores/{lang}/listener_scores.json','w') as file:
        json.dump(listener_scores, file)

def get_scores_from_multiple(lang):
    with open(f'data/{lang}.json','r') as file:
        dataset = json.load(file)

    speaker_scores = defaultdict(list)
    finger_ind = defaultdict(list)
    listener_scores = defaultdict(list)

    for sample in dataset['samples']:            
        speaker_scores[sample['letter']].append(distance(sample, resting))
        finger_ind[sample['letter']].append(finger_independence(sample))
        for sample2 in dataset['samples']:
            char1, char2 = sample['letter'], sample2['letter']
            if char1 != char2:
                listener_scores[char1+char2 if char1 <= char2 else char2+char1].append(distance(sample, sample2))

    if not os.path.exists(f'scores/{lang}'):
        # Create the directory
        os.makedirs(f'scores/{lang}')

    with open(f'scores/{lang}/speaker_all.json','w') as file:
        json.dump(speaker_scores, file)
    with open(f'scores/{lang}/finger_ind_all.json','w') as file:
        json.dump(finger_ind, file)
    with open(f'scores/{lang}/listener_all.json','w') as file:
        json.dump(listener_scores, file)


def main():
    for lang in ['dgs', 'fsl', 'jsl2', 'psl']: # ['afb', 'ase', 'isg', 'jsl']:
        get_scores(lang)

if __name__ == '__main__':
    main()