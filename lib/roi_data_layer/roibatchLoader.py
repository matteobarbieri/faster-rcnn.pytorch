
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch

from scipy.misc import imsave

from torchvision.transforms import ColorJitter
from PIL import ImageDraw, Image, ImageEnhance

import numpy as np

import os
import cv2


def _apply_sp(img, amount, point_size=1, s_vs_p=0.2, salt_color=255, pepper_color=0):
    noisy = np.copy(img)
    point_size = max(1, point_size)
    rectangle = np.ones(shape=(point_size, point_size))
    rect_idx = [axis_idx - int(point_size / 2) for axis_idx in np.where(rectangle)]

    # Salt mode
    num_salt = np.ceil(amount * np.prod(img.shape[:2]) * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape[:2]]
    coords = [np.clip(c[:, None] + r, 0, d - 1).reshape(-1) for c, d, r in zip(coords, img.shape[:2], rect_idx)]
    noisy[tuple(coords)] = salt_color

    # Pepper mode
    num_pepper = np.ceil(amount * np.prod(img.shape[:2]) * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape[:2]]
    coords = [np.clip(c[:, None] + r, 0, d - 1).reshape(-1) for c, d, r in zip(coords, img.shape[:2], rect_idx)]
    noisy[tuple(coords)] = pepper_color

    return noisy


# def noisy(noise_typ,image):
   # if noise_typ == "gauss":
      # row,col,ch= image.shape
      # mean = 0
      # var = 0.1
      # sigma = var**0.5
      # gauss = np.random.normal(mean,sigma,(row,col,ch))
      # gauss = gauss.reshape(row,col,ch)
      # noisy = image + gauss
      # return noisy
   # elif noise_typ == "s&p":
      # row,col,ch = image.shape
      # s_vs_p = 0.5
      # amount = 0.004
      # out = np.copy(image)
      # # Salt mode
      # num_salt = np.ceil(amount * image.size * s_vs_p)
      # coords = [np.random.randint(0, i - 1, int(num_salt))
              # for i in image.shape]
      # out[coords] = 1

      # # Pepper mode
      # num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      # coords = [np.random.randint(0, i - 1, int(num_pepper))
              # for i in image.shape]
      # out[coords] = 0
      # return out
   # elif noise_typ == "poisson":
      # vals = len(np.unique(image))
      # vals = 2 ** np.ceil(np.log2(vals))
      # noisy = np.random.poisson(image * vals) / float(vals)
      # return noisy
   # elif noise_typ =="speckle":
      # row,col,ch = image.shape
      # gauss = np.random.randn(row,col,ch)
      # gauss = gauss.reshape(row,col,ch)
      # noisy = image + image * gauss
      # return noisy

class roibatchLoader(data.Dataset):

    def __init__(self, roidb, ratio_list, ratio_index, batch_size,
                 num_classes, training=True, normalize=None,
                 transform=None):

        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)
        self.transform = transform

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

    def __getitem__(self, index):  # noqa
        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        # print("type blobs data" + str(type(blobs['data'])))
        # print("blobs data shape" + str(blobs['data'].shape))
        # print("here" + str(type(data)))
        # print("blobs data max: {}".format(blobs['data'].max()))
        # print("blobs data min: {}".format(blobs['data'].min()))
        # print("blobs data std: {}".format(blobs['data'].std()))

        # noisy_data_imarray = blobs['data'] + \
            # np.random.normal(0, 10, size=blobs['data'].shape)

        image = blobs['data'][0]

        if self.training:
            # Backto RGB
            image = image[:,:,::-1]

            # imsave('/home/matteo/a1.png', image)

            salt_color = image.min()
            pepper_color = image.max()

            # print("salt color: {}".format(salt_color))
            # print("pepper color: {}".format(pepper_color))

            # image2 = image * 0.9
            # image3 = image * 1.1

            # image2 = enhancer.enhance(0.9)
            # image3 = enhancer.enhance(1.1)

            # imsave('/home/matteo/a2.png', image2)
            # imsave('/home/matteo/a3.png', image3)

            NOISE_SP_AMOUNT = (0.0005, 0.001)

            if np.random.random() < 0.5:
            # Add salt and pepper effect
                image = _apply_sp(
                    image,
                    np.random.uniform(*NOISE_SP_AMOUNT),
                    point_size=2,
                salt_color=salt_color,
                pepper_color=pepper_color)

            # print(image.shape)
            # imsave('/home/matteo/a4.png', image)

            # if np.random.random() < 0.5:

                # data = noisy('s&p', blobs['data'])


            # print("Shape: {}".format(blobs['data'][0].shape))
            # print("Shape: {}".format(noisy_data_imarray.shape))

            # imsave('/home/matteo/a1.png', blobs['data'][0])
            # imsave('/home/matteo/a2.png', noisy_data_imarray[0])
            # imsave('/home/matteo/a2.png', noisy_data_imarray)

            # data = torch.from_numpy(blobs['data'])

            # Backto BGR
            image = image[:,:,::-1]

        data = torch.from_numpy(image[np.newaxis, :, :, :].copy())

        # 1/0
        im_info = torch.from_numpy(blobs['im_info'])
        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)
        if self.training:
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both
            # conditions. (done)
            # NOTE2: need to consider the situation for the tail samples.
            # (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:

                if ratio < 1:

                    # this means that data_width << data_height, we need to
                    # crop the data_height
                    min_y = int(torch.min(gt_boxes[:, 1]))
                    max_y = int(torch.max(gt_boxes[:, 3]))

                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(
                                    range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region-trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(
                                    range(min_y, min_y+y_s_add))

                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordiante of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bounding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                else:
                    # this means that data_width >> data_height, we need to
                    # crop the data_width
                    min_x = int(torch.min(gt_boxes[:, 0]))
                    max_x = int(torch.max(gt_boxes[:, 2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            x_s_min = max(max_x-trim_size, 0)
                            x_s_max = min(min_x, data_width-trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(
                                    range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region-trim_size)/2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(
                                    range(min_x, min_x+x_s_add))

                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)

            # based on the ratio, padding the image.
            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.FloatTensor(
                    int(np.ceil(data_width / ratio)), data_width, 3).zero_()

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(
                    data_height, int(np.ceil(data_height * ratio)), 3).zero_()

                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(
                    trim_size, trim_size, 3).zero_()

                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size

            # check the bounding box:
            not_keep = (gt_boxes[:, 0] == gt_boxes[:, 2]) | \
                       (gt_boxes[:, 1] == gt_boxes[:, 3])

            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.FloatTensor(
                self.max_num_box, gt_boxes.size(1)).zero_()
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes, :] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0

                # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)

            # Apply transformation for data augmentation
            if self.transform is not None:
                padding_data = self.transform(padding_data)

            # print("here2" + str(type(padding_data)))
            return padding_data, im_info, gt_boxes_padding, num_boxes
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(
                3, data_height, data_width)

            im_info = im_info.view(3)

            gt_boxes = torch.FloatTensor([1, 1, 1, 1, 1])
            num_boxes = 0

            # Apply transformation for data augmentation
            if self.transform is not None:
                data = self.transform(data)

            return data, im_info, gt_boxes, num_boxes

    def __len__(self):
        return len(self._roidb)
