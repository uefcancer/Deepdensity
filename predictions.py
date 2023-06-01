import os
import argparse
import torch
import torch.nn as nn
from dataset import MammoEvaluation
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='MG_Tuni', type=str, help='dataset ')
    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='mg_tuni_unet_aug.pth', type=str,
                       help='path to save the model')  # change here
    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)

#test_dataset = MammoEvaluation(path=config['data_path'])
test_dataset = MammoEvaluation(path='data/dataset_name/valid/input_image/*')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)


for i, (img_id, img_shape,  image) in enumerate(test_dataloader):
    #print(i, image.shape)
    image = image.cuda()
    pred1, pred2 = model.module.predict(image)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    image = image[:, :, 0]
    pred1 = pred1[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred1 = pred1[:, :, 0]

    pred2 = pred2[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred2 = pred2[:, :, 0]

    image = cv2.resize(image, (img_shape[1].item(), img_shape[0].item()))
    pred1 = cv2.resize(pred1, (img_shape[1].item(), img_shape[0].item()))
    pred2 = cv2.resize(pred2, (img_shape[1].item(), img_shape[0].item()))

    plt.imsave(os.path.join('test_output/unet_aug_predictions/images', img_id[0]), image, cmap='gray')
    plt.imsave(os.path.join('test_output/unet_aug_predictions/breast_masks', img_id[0]), pred1, cmap='gray')
    plt.imsave(os.path.join('test_output/unet_aug_predictions/dense_masks', img_id[0]), pred2, cmap='gray')

    #plot_images(image, pred1, pred2)
    print(i, img_id[0], image.shape, pred1.shape, pred2.shape)














