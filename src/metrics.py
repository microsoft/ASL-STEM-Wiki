# From https://github.com/chevalierNoir/FS-Detection/blob/main/src/metrics/ap_iou.py

import os
import pickle
import numpy as np
from sklearn.cluster import KMeans


def get_iou(y_pred, y_true):
    # y_pred, y_true: [[start, end],...]
    # return: ndarray of shape (n_pred, n_true)
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    n_pred, n_true = y_pred.shape[0], y_true.shape[0]
    y_pred = np.repeat(y_pred.reshape(n_pred, 1, 2), n_true, axis=1).reshape(-1, 2)
    y_true = np.repeat(y_true.reshape(1, n_true, 2), n_pred, axis=0).reshape(-1, 2)
    max_start, min_end = np.maximum(y_pred[:, 0], y_true[:, 0]), np.minimum(y_pred[:, 1], y_true[:, 1])
    min_start, max_end = np.minimum(y_pred[:, 0], y_true[:, 0]), np.maximum(y_pred[:, 1], y_true[:, 1])
    intersection = min_end - max_start + 1
    union = max_end - min_start + 1
    iou = (intersection / union).reshape(n_pred, n_true).clip(min=0)
    iou = iou.reshape(n_pred, n_true)
    return iou


def get_precision_recall(y_pred, y_true, thr, ordering=False):
    n_pred, n_true = len(y_pred), len(y_true)
    if n_pred == 0:
        precision_tp, recall_tp = 0, 0
    else:
        iou = get_iou(y_pred, y_true)
        conf = (iou > thr).astype(np.int32)
        # if >1 detections for one segment, only the one with highest IoU is TP
        if ordering:
            n_tp = 0
            for i in range(n_pred):
                max_id = np.argmax(iou[i])
                if conf[i, max_id]:
                    n_tp += 1
                    iou, conf = np.delete(iou, max_id, axis=1), np.delete(conf, max_id, axis=1)
                if iou.shape[1] == 0:
                    break
            precision_tp, recall_tp = n_tp, n_tp
        else:
            try:
                mask = (np.max(iou, axis=0).reshape(1, -1) == iou).astype(np.int32)
            except Exception as err:
                print(iou)
                raise err
            conf = conf * mask
            precision_tp, recall_tp = (conf.sum(axis=1) > 0).sum(), (conf.sum(axis=0) > 0).sum()
    return precision_tp, recall_tp, n_pred, n_true


def get_mAP(y_pred, y_true, stat_pkl, iou_thrs=[0.1], ordering=False):
    num_roi_thrs = range(1, 50)
    aps = {}
    for iou_thr in iou_thrs:
        precisions, recalls = [], []
        for num_roi_thr in num_roi_thrs:
            y_pred_i = [x[:num_roi_thr, :2] for x in y_pred]
            # parallel
            args = list(zip(y_pred_i, y_true, [iou_thr for _ in range(len(y_pred_i))], [ordering for _ in range(len(y_pred_i))]))
            prs = list(map(lambda x: get_precision_recall(*x), args))
            prs = list(zip(*prs))
            precision_tp, recall_tp, n_pred, n_true = sum(prs[0]), sum(prs[1]), sum(prs[2]), sum(prs[3])
            precision, recall = precision_tp / max(n_pred, 1), recall_tp / max(n_true, 1)
            precisions.append(precision)
            recalls.append(recall)
        print(f"IoU={iou_thr}")
        aps[iou_thr] = [precisions, recalls]
    pickle.dump(aps, open(stat_pkl, 'wb'))
    return aps

def compute_mAP_from_stat(stat_pkl, ptype='pascal'):
    ss = pickle.load(open(stat_pkl, 'rb'))
    if ptype == 'pascal':
        target_recall_vals = np.arange(0, 1.05, 0.1)
    elif ptype == 'coco':
        target_recall_vals = np.arange(0, 1.00001, 0.01)
    else:
        raise NotImplementedError
    err_thr = 0.1
    iou_to_mAP = {}
    for iou_thr, stats in ss.items():
        precision, recall = np.array(stats[0]), np.array(stats[1])
        target_precision_vals = []
        for val in target_recall_vals:
            recall_diff = recall - val
            valid_ids = recall_diff >= 0
            if valid_ids.sum() > 0:
                target_precision_val = np.max(precision[valid_ids])
            else:
                target_precision_val = 0
            target_precision_vals.append(target_precision_val)
        mAP = np.array(target_precision_vals).mean()
        iou_to_mAP[iou_thr] = mAP
        print('IoU %.1f, mAP: %.3f' % (iou_thr, mAP))
    return iou_to_mAP

def get_precision_per_sample(pred_pkl, stat_pkl, iou_thr=0.5, recall_thr=0.5):
    ss = pickle.load(open(pred_pkl, 'rb'))
    y_true = ss['grt']
    num_roi_thrs = range(1, 50)
    precision_all = []
    for i in range(len(y_true)):
        precisions, recalls = np.zeros(len(num_roi_thrs)), np.zeros(len(num_roi_thrs))
        for j, num_roi_thr in enumerate(num_roi_thrs):
            y_pred_i = [ss['pred'][i][:num_roi_thr, :2]]
            y_true_i = [ss['grt'][i]]
            # parallel
            args = list(zip(y_pred_i, y_true_i, [iou_thr for _ in range(len(y_pred_i))]))
            prs = list(map(lambda x: get_precision_recall(*x), args))
            prs = list(zip(*prs))
            precision_tp, recall_tp, n_pred, n_true = sum(prs[0]), sum(prs[1]), sum(prs[2]), sum(prs[3])
            precision, recall = precision_tp / max(n_pred, 1), recall_tp / max(n_true, 1)
            precisions[j] = precision
            recalls[j] = recall
        mask = recalls >= recall_thr
        if mask.sum() > 0:
            precision_val = precisions[mask].max()
        else:
            precision_val = 0
        precision_all.append(precision_val)
    pickle.dump(precision_all, open(stat_pkl, 'wb'))
    return 0

def get_ap_iou(y_pred, y_true, pred_pkl, iou_thrs=[0.1, 0.2, 0.3, 0.4, 0.5], ptype='coco'):
    stat_pkl, result_txt = pred_pkl + ".tmp", pred_pkl + '.iou'
    get_mAP(y_pred, y_true, pred_pkl, stat_pkl, iou_thrs=iou_thrs, ordering=True)
    iou_to_mAP = compute_mAP_from_stat(stat_pkl, ptype)
    print(f"Write AP@IoU into {result_txt}")
    with open(result_txt, 'w') as fo:
        for iou, mAP in iou_to_mAP.items():
            fo.write(str(iou)+","+str(mAP)+"\n")
    os.remove(stat_pkl)
    return iou_to_mAP

def merge_spans(spans):
    if not spans:
        return []

    # Sort the spans based on their start values
    spans.sort(key=lambda x: x[0])

    merged_spans = [spans[0]]

    for current_start, current_end, current_score in spans[1:]:
        previous_start, previous_end, previous_score = merged_spans[-1]

        if current_start <= previous_end:
            # Overlapping spans, update the previous span's end value
            merged_spans[-1] = [previous_start, max(current_end, previous_end), max(current_score, previous_score)]
        else:
            # Non-overlapping span, add it to the merged_spans list
            merged_spans.append([current_start, current_end, current_score])

    return merged_spans

def cosine_similarity(vector1, vector2):
    vector1, vector2 = vector1.flatten(), vector2.flatten()
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity


def get_iou(pred_spans, true_spans):
    def get_intersection(span1, span2):
        # Calculate the intersection between two spans
        start = max(span1[0], span2[0])
        end = min(span1[1], span2[1])
        return max(0, end - start)

    def get_union(span1, span2, intersection):
        # Calculate the union between two spans
        return (span1[1] - span1[0]) + (span2[1] - span2[0]) - intersection

    iou_scores = []

    for true_span in true_spans:
        max_iou = 0
        for pred_span in pred_spans:
            intersection = get_intersection(true_span, pred_span)
            union = get_union(true_span, pred_span, intersection)
            iou = intersection / union if union > 0 else 0
            max_iou = max(max_iou, iou)
        iou_scores.append(max_iou)

    return iou_scores

def calculate_iou(span1, span2):
    # Calculate Intersection over Union (IoU) between two spans
    x1, x2 = max(span1[0], span2[0]), min(span1[1], span2[1])
    intersection = max(0, x2 - x1 + 1)
    union = (span1[1] - span1[0] + 1) + (span2[1] - span2[0] + 1) - intersection
    iou = intersection / union
    return iou

def compute_ap_at_iou(y_pred, y_true, iou_thresholds):
    # Sort the predicted spans in descending order based on confidence scores
    y_pred.sort(key=lambda x: x[2], reverse=True)

    # Initialize variables to store True Positive (TP), False Positive (FP), and False Negative (FN) counts for each threshold
    tp_counts = {iou_threshold: 0 for iou_threshold in iou_thresholds}
    fp_counts = {iou_threshold: 0 for iou_threshold in iou_thresholds}
    fn_counts = {iou_threshold: len(y_true) for iou_threshold in iou_thresholds}

    # Iterate through predicted spans and compare with ground truth spans to calculate TP, FP, and FN counts
    for pred_span in y_pred:
        pred_start, pred_end, confidence = pred_span
        matched_iou = False

        for iou_threshold in iou_thresholds:
            for true_span in y_true:
                true_start, true_end = true_span

                if calculate_iou((pred_start, pred_end), (true_start, true_end)) >= iou_threshold:
                    tp_counts[iou_threshold] += 1
                    fn_counts[iou_threshold] -= 1
                    matched_iou = True
                    break

            if not matched_iou:
                fp_counts[iou_threshold] += 1

    # Calculate Precision and Recall for each threshold
    try:
        precisions = {iou_threshold: tp_counts[iou_threshold] / (tp_counts[iou_threshold] + fp_counts[iou_threshold])
                    for iou_threshold in iou_thresholds}
        recalls = {iou_threshold: tp_counts[iou_threshold] / (tp_counts[iou_threshold] + fn_counts[iou_threshold])
                for iou_threshold in iou_thresholds}
    except:
        return 0

    # Calculate Average Precision (AP) for each threshold
    ap_values = []
    for iou_threshold in iou_thresholds:
        if tp_counts[iou_threshold] > 0:
            ap = sum(precisions[iou_threshold] for iou_thr in iou_thresholds if recalls[iou_thr] >= recalls[iou_threshold]) / len(iou_thresholds)
            ap_values.append(ap)

    # Compute the Average Precision at IoU (AP@IOU)
    try:
        ap_at_iou = sum(ap_values) / len(ap_values)
    except:
        ap_at_iou = 0

    return ap_at_iou

def find_best_clustering(scores, num_clusters):
    # Convert the scores list to a numpy array
    scores_array = np.array(scores).reshape(-1, 1)

    # Initialize the KMeans object with the desired number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10, max_iter=1000)

    # Fit the KMeans model to the data
    kmeans.fit(scores_array)

    # Get the cluster assignments for each data point
    cluster_assignments = kmeans.labels_

    # Initialize a list to store the smallest value in each cluster
    smallest_in_clusters = []

    # Iterate over each cluster and find the smallest value
    for cluster_idx in range(num_clusters):
        cluster_values = scores_array[cluster_assignments == cluster_idx]
        if len(cluster_values) == 0:
            smallest_value = np.inf
        else:
            smallest_value = np.min(cluster_values)
        smallest_in_clusters.append(smallest_value)

    return smallest_in_clusters