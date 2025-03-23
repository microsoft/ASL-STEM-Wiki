import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import csv
from timeit import default_timer as timer
import os
import multiprocessing
import fire

def worker(f, save_path):
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False, min_detection_confidence=0.5) as holistic:
        
        video = cv2.VideoCapture(f)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        feature = np.zeros((int(total_frames), 543, 2))
        count = 0
        success = 1

        while success: 
            success, image = video.read()            
            if success:
                try:
                    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    for i in range(33):
                        if results.pose_landmarks:
                            feature[count][i][0] = results.pose_landmarks.landmark[i].x
                            feature[count][i][1] = results.pose_landmarks.landmark[i].y

                    j = 33
                    for i in range(21):
                        if results.right_hand_landmarks:
                            feature[count][i+j][0] = results.right_hand_landmarks.landmark[i].x
                            feature[count][i+j][1] = results.right_hand_landmarks.landmark[i].y

                    j = 54
                    for i in range(21):
                        if results.left_hand_landmarks:
                            feature[count][i+j][0] = results.left_hand_landmarks.landmark[i].x
                            feature[count][i+j][1] = results.left_hand_landmarks.landmark[i].y

                    j = 75
                    for i in range(468):
                        if results.face_landmarks:
                            feature[count][i+j][0] = results.face_landmarks.landmark[i].x
                            feature[count][i+j][1] = results.face_landmarks.landmark[i].y
                    count += 1
                except Exception as e:
                    print(e)

        print('About to save:', save_path)
        np.save(save_path, feature)
        
def main(num_shards=1, shard_index=0):
    # Initialize mediapipe drawing class - to draw the landmarks points.
    mp_drawing = mp.solutions.drawing_utils

    #Update paths here
    src_path = 'ASL_STEM_Wiki_data/videos'
    dst_path = 'ASL_STEM_Wiki_data/keypoints'
    data_csv  = 'fs-annotations/all.csv'

    import pandas as pd

    df = pd.read_csv(data_csv)

    count_f = 0
    start = timer()

    pool = multiprocessing.Pool(128)

    for i,row in df.iterrows():
        if i % num_shards != shard_index:
            continue
        
        video_path = row.iloc[4]
        f = os.path.join(src_path, video_path) #### filename
        
        name = ''.join(video_path[:-4].split('/'))
        save_path = os.path.join(dst_path, name + '.npy')
        
        print(f'[{i}]', name, save_path)
        
        if os.path.exists(save_path):
            print('File already exists:', save_path)
            continue

        pool.apply_async(worker, args=(f, save_path))
        worker(f, save_path)

if __name__ == '__main__':
    fire.Fire(main)