import os
import argparse
import torch
import torch.nn as nn
from dataset import MammoEvaluation
from torch.utils.data import DataLoader
import segmentation_models_multi_tasking as smp
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='dataset_name', type=str, help='dataset ')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='test_output/models/unet.pth', type=str, help='path to save the model') # change her
 
    
    parser.add_argument('--results_path', default='test_output/test/report.txt', type=str, help='path to save the model')  # change here

    config = parser.parse_args()
    return config

config = vars(parse_args())
#print(config)

#test_dataset = MammoEvaluation(path=config['data_path'])
test_dataset = MammoEvaluation(path=config['data_path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)
# load old saved model
model_old = torch.load("./models/base.pth")
model_old = nn.DataParallel(model_old.module)

b_precision_values = []
b_recall_values = []
b_fscore_values = []
b_accuracy_values = []
b_iou_values = []
d_precision_values = []
d_recall_values = []
d_fscore_values = []
d_accuracy_values = []
d_iou_values = []
old_dense_values = []
new_dense_values = []
diff_dense_values = []

IOU = smp.utils.metrics.IoU()
Precision = smp.utils.metrics.Precision()
Recall = smp.utils.metrics.Recall()
Accuracy = smp.utils.metrics.Accuracy()
Fscore = smp.utils.metrics.Fscore()

def average_count(data):
    mean = np.mean(data)
    mean = round(mean, 3)
    ci_min, ci_max = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, round((ci_max - ci_min), 3)



with open(config['results_path'], 'a+') as logs_file, open('test_output/test/report_density_difference.txt', 'a+') as density_file:
    logs_file.seek(0)
    existing_content = logs_file.read().strip()
    
    density_file.seek(0)
    dense_existing_content = density_file.read().strip()
    
    if not existing_content or not existing_content.startswith('Abbreviations') and not dense_existing_content or not existing_content.startswith('Image_ID'):

        abbreviations = "Abbreviations\nB = Breast\nD = Dense\nPrec = Precision \nRec = Recall\nF-Sc = Fscore\nAcc = Accuracy\n-------------------"
        header = "Image_ID\t\t\t\tB_Prec\tB_Rec\tB_F-Sc\tB_Acc\tB_IoU\tD_Prec\tD_Rec\tD_F-Sc\tD_Acc\tD_IoU"
        logs_file.write(abbreviations + '\n')
        logs_file.write(header + '\n')
        dense_header = "Image_ID\t\t\t\tPredicted Density\t\tGround Truth\t\tDifference"
        density_file.write(dense_header + '\n')
    
    for (img_id, image, b_mask_org, d_mask_org) in tqdm(test_dataloader):
        image = image.cuda()
        b_mask_org = b_mask_org.cuda()
        d_mask_org = d_mask_org.cuda()

        pred_b_mask, pred_d_mask = model.module.predict(image)
        pred_old_b_mask, pred_old_d_mask = model_old.module.predict(image)

        breast_iou = IOU(pred_b_mask, b_mask_org)
        breast_precision = Precision(pred_b_mask, b_mask_org)
        breast_recall = Recall(pred_b_mask, b_mask_org)
        breast_accuracy = Accuracy(pred_b_mask, b_mask_org)
        breast_fscore = Fscore(pred_b_mask, b_mask_org)

        dense_iou = IOU(pred_d_mask, d_mask_org)
        dense_precision = Precision(pred_d_mask, d_mask_org)
        dense_recall = Recall(pred_d_mask, d_mask_org)
        dense_accuracy = Accuracy(pred_d_mask, d_mask_org)
        dense_fscore = Fscore(pred_d_mask, d_mask_org)
        
        breast_iou = round(breast_iou.item(), 3)
        breast_precision = round(breast_precision.item(), 3)
        breast_recall = round(breast_recall.item(), 3)
        breast_accuracy = round(breast_accuracy.item(), 3)
        breast_fscore = round(breast_fscore.item(), 3)

        dense_iou = round(dense_iou.item(), 3)
        dense_precision = round(dense_precision.item(), 3)
        dense_recall = round(dense_recall.item(), 3)
        dense_accuracy = round(dense_accuracy.item(), 3)
        dense_fscore = round(dense_fscore.item(), 3)

        b_precision_values.append(breast_precision)
        b_recall_values.append(breast_recall)
        b_fscore_values.append(breast_fscore)
        b_accuracy_values.append(breast_accuracy)
        b_iou_values.append(breast_iou)

        d_precision_values.append(dense_precision)
        d_recall_values.append(dense_recall)
        d_fscore_values.append(dense_fscore)
        d_accuracy_values.append(dense_accuracy)
        d_iou_values.append(dense_iou)
            
        row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(img_id[0],
                                            breast_precision,
                                            breast_recall,
                                            breast_fscore,
                                            breast_accuracy,
                                            breast_iou,
                                            dense_precision,
                                            dense_recall,
                                            dense_fscore,
                                            dense_accuracy,
                                            dense_iou)
        print(row, file=logs_file)
        
        pred_b_mask = pred_b_mask[0].cpu().numpy().transpose(1, 2, 0)
        pred_b_mask = pred_b_mask[:, :, 0]

        pred_d_mask = pred_d_mask[0].cpu().numpy().transpose(1, 2, 0)
        pred_d_mask = pred_d_mask[:, :, 0]

        pred_old_b_mask = pred_old_b_mask[0].cpu().numpy().transpose(1, 2, 0)
        pred_old_b_mask = pred_old_b_mask[:, :, 0]

        pred_old_d_mask = pred_old_d_mask[0].cpu().numpy().transpose(1, 2, 0)
        pred_old_d_mask = pred_old_d_mask[:, :, 0]

        breast_area = np.sum(np.array(pred_b_mask) == 1)
        dense_area = np.sum(np.array(pred_d_mask) == 1)
        new_density = round(((dense_area / breast_area) * 100), 3)
        #print("NEW \n Density: ", round(new_density, 3))

        old_breast_area = np.sum(np.array(pred_old_b_mask) == 1)
        old_dense_area = np.sum(np.array(pred_old_d_mask) == 1)
        old_density = round(((old_dense_area / old_breast_area) * 100), 3)
        #print("OLD \n Density: ", round(old_density,3))

        diff = round(abs(new_density - old_density), 3)
        #print("Difference is = ", diff)

        old_dense_values.append(old_density)
        new_dense_values.append(new_density)
        diff_dense_values.append(diff)

        dense = '{}\t{}\t\t\t{}\t\t\t{}'.format(img_id[0],
                                            new_density,
                                            old_density,
                                            diff)
        print(dense, file=density_file)

        #break

    b_iou_mean, b_iou_ci = average_count(b_iou_values)
    b_precision_mean, b_precision_ci = average_count(b_precision_values)
    b_recall_mean, b_recall_ci = average_count(b_recall_values)
    b_accuracy_mean, b_accuracy_ci = average_count(b_accuracy_values)
    b_fscore_mean, b_fscore_ci = average_count(b_fscore_values)
    
    d_iou_mean, d_iou_ci = average_count(d_iou_values)
    d_precision_mean, d_precision_ci = average_count(d_precision_values)
    d_recall_mean, d_recall_ci = average_count(d_recall_values)
    d_accuracy_mean, d_accuracy_ci = average_count(d_accuracy_values)
    d_fscore_mean, d_fscore_ci = average_count(d_fscore_values)

    avg_ci_row = '\nAverage\t\t\t\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\nConfidance Interval\t\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(b_precision_mean,
                                                                               b_recall_mean,
                                                                               b_fscore_mean,
                                                                               b_accuracy_mean,
                                                                               b_iou_mean,
                                                                               d_precision_mean,
                                                                               d_recall_mean,
                                                                               d_fscore_mean,
                                                                               d_accuracy_mean,
                                                                               d_iou_mean,
                                                                               b_precision_ci,
                                                                               b_recall_ci,
                                                                               b_fscore_ci,
                                                                               b_accuracy_ci,
                                                                               b_iou_ci,
                                                                               d_precision_ci,
                                                                               d_recall_ci,
                                                                               d_fscore_ci,
                                                                               d_accuracy_ci,
                                                                               d_iou_ci)
    
    print(avg_ci_row, file=logs_file)
    old_dense_mean, old_dense_ci = average_count(old_dense_values)
    new_dense_mean, new_dense_ci = average_count(new_dense_values)
    diff_dense_mean, diff_dense_ci = average_count(diff_dense_values)

    avg_dense_mean_ci = '\nAverage\t\t\t\t\t{}\t\t\t{}\t\t\t{}\nConfidance Interval\t\t{}\t\t\t{}\t\t\t{}'.format(old_dense_mean,
                                                                            new_dense_mean,
                                                                            diff_dense_mean,
                                                                            old_dense_ci,
                                                                            new_dense_ci,
                                                                            diff_dense_ci)
    
    print(avg_dense_mean_ci, file=density_file)
