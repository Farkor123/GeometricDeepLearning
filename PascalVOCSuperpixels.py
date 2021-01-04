import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from torch_geometric.data import Data, DataLoader
import torch

import sys
import xml.etree.ElementTree as ET

feature_dict = {'motorbike': 0,
                'bus': 1,
                'bicycle': 2,
                'chair': 3,
                'horse': 4,
                'pottedplant': 5,
                'tvmonitor': 6,
                'diningtable': 7,
                'cat': 8,
                'train': 9,
                'sheep': 10,
                'background': 11,
                'bottle': 12,
                'person': 13,
                'boat': 14,
                'car': 15,
                'dog': 16,
                'bird': 17,
                'aeroplane': 18,
                'sofa': 19,
                'cow': 20}

if __name__ == '__main__':
    path = '/home/farkor123/Downloads/VOC_files/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    data_list = []
    f = open(path + '/ImageSets/Segmentation/train.txt', 'r')
    for line in f:
        image_path = path + '/JPEGImages/' + line[:-1] + '.jpg'
        annotations_path = path + '/Annotations/' + line[:-1] + '.xml'

        image = img_as_float(imread(image_path))
        segmentation = slic(image, n_segments=250, compactness=10, sigma=1, start_label=0)
        segments_count = len(np.unique(segmentation))
        superpixels_position = np.zeros((segments_count, 3))
        superpixels_color = np.zeros((segments_count, 4))
        superpixels_target = [11] * segments_count
        img_processed = np.zeros((image.shape[0], image.shape[1], 3))

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                q = segmentation[x, y]
                superpixels_position[q] = np.add(superpixels_position[q], [x, y, 1])
                superpixels_color[q] = np.add(superpixels_color[q], [i for i in image[x, y]] + [1])
        superpixels_color = np.array(
            [[pixel[0]*100 // pixel[3], pixel[1]*100 // pixel[3], pixel[2]*100 // pixel[3]] for pixel in superpixels_color])
        superpixels_position = np.array(
            [[pixel[0] // pixel[2], pixel[1] // pixel[2]] for pixel in superpixels_position])

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                img_processed[x, y] = superpixels_color[segmentation[x, y]]

        edge_index = [[], []]
        for i in range(segments_count - 1):
            if i < segments_count - 1:
                for j in range(i + 1, segments_count):
                    edge_index[0].append(i)
                    edge_index[1].append(j)

        annotations_tree = ET.parse(annotations_path)
        root = annotations_tree.getroot()
        for boxes in root.iter('object'):
            ymin, xmin, ymax, xmax, name, pose = None, None, None, None, None, None

            ymin = int(boxes.find('bndbox/ymin').text)
            xmin = int(boxes.find('bndbox/xmin').text)
            ymax = int(boxes.find('bndbox/ymax').text)
            xmax = int(boxes.find('bndbox/xmax').text)
            name = boxes.find('name').text
            pose = boxes.find('pose').text

            for i in range(segments_count):
                if superpixels_position[i][0] > xmin and superpixels_position[i][0] < xmax and superpixels_position[i][1] > ymin and superpixels_position[i][1] < ymax:
                    superpixels_target[i] = feature_dict[name]

        data = Data(x=torch.LongTensor(superpixels_color), edge_index=torch.LongTensor(edge_index), pos=torch.LongTensor(superpixels_position), y=torch.LongTensor(superpixels_target))
        data_list.append(data)
    torch.save(data_list, '250_SuperPixels.pt')
    print("DONE")


    '''
        f = plt.figure()
        f.add_subplot(1, 3, 1)
        plt.imshow(image)
        f.add_subplot(1, 3, 2)
        plt.imshow(mark_boundaries(image, segmentation))
        f.add_subplot(1, 3, 3)
        plt.imshow(img_processed)
        plt.show(block=True)
    '''
