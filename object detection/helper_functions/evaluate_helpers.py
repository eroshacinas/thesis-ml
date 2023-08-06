def yolo_format_to_bbox_corners(df):  # [x1, y1, w, h]
    w = df[:, 2] / 2
    h = df[:, 3] / 2

    # make yolo labels format to corners of the bboxes
    df[:, 0], df[:, 2] = df[:, 0] - w, df[:, 0] + w
    df[:, 1], df[:, 3] = df[:, 1] - h, df[:, 1] + h

    return df
    
def _evaluate(gt_df, pred_df, target_class, iou_thres=0.5):
    import glob
    import pandas as pd
    import numpy as np
    import os
    import torch
    from torchvision import ops
    import shutil
    
    
    # convert yolo format to bbox corners
    gt_tensor = torch.from_numpy(yolo_format_to_bbox_corners(gt_df.values[:, 1:])) # convert xywh to xyxy, convert to tensor
    pred_tensor = torch.from_numpy(yolo_format_to_bbox_corners(pred_df.values[:, 1:]))

    iou_tensor = ops.box_iou(gt_tensor, pred_tensor) # get iou tensor
    max_indices = torch.argmax(iou_tensor, axis=1, keepdim=False).numpy() # get max indices of each row
    
    
    # get true positive, false positive, false negative
    TP, FP, FN = 0, 0, 0
    iou_threshold = 0.5

    for i, key in enumerate(max_indices):
        #if(iou_tensor[i, key] > iou_threshold) and (gt_df["class"][i] == pred_df["class"][key]): # if iou is greater than threshold and class is same
        if(iou_tensor[i, key] > iou_threshold) and (gt_df["class"][i] == target_class) and (pred_df["class"][key]==target_class): 
            TP += 1

        elif(iou_tensor[i, key] < iou_threshold) and (gt_df["class"][i] == target_class) and (pred_df["class"][key]==target_class):
            FN += 1 # if iou is less than threshold or class is different
            
    
    # get false positive
    #FP = pred_df.shape[0] - TP
    FP = (pred_df['class'] == target_class).sum() - TP
    
    return TP, FP, FN

def _precision_recall_f1(TP, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = (2*precision*recall) / (precision + recall)
    return precision, recall, f1
