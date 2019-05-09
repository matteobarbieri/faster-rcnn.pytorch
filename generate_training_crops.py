#!/usr/bin/env python
from PIL import Image, ImageDraw

from tqdm import tqdm

import os, sys

import random
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(
            description='Crops target image for inference.'
    )

    parser.add_argument(
            '-v', 
            '--verbose', 
            action='store_true', 
            help='Verbose'
    )
    
    # Output folder path
    parser.add_argument(
            'dataset_folder', 
            type=str, 
            help="Folder where the dataset is"
    )

    # Output folder path
    parser.add_argument(
            'output_folder', 
            type=str, 
            help="Folder where the output will be saved"
    )

    # Output folder path
    parser.add_argument(
            '--crop_size',
            type=int,
            default=600,
            help="The size of the crop (square, default 600)"
    )

    # Output folder path
    parser.add_argument(
            '--crop_stride',
            type=int,
            default=200,
            help="The stride of the sliding window (default 200)"
            )

    # Output folder path
    parser.add_argument(
            '--n_crops',
            type=int,
            default=20,
            help="The number of (valid) random crops for each input image"
            )

    args = parser.parse_args()

    return args

def get_intersection(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] <= bb1[1]
    assert bb1[2] <= bb1[3]
    assert bb2[0] <= bb2[1]
    assert bb2[2] <= bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    return intersection_area

def do_crop(synth_diagram, label_lines, output_id, out_folder):
    """

    Parameters
    ----------

    label_lines : list
        Each element is a 5 tuple: class_id, x, y, w, h

    output_patch_name : str
        Name of the output patch

    """

    # Define fixed crop window size
    CROP_W = CROP_H = 608

    # Treshold for deciding wether to include a label in a crop
    INTERSECTION_THRESHOLD = 0.4
    
    # Sample random
    d_w, d_h = synth_diagram.size

    # Keep going until you get a crop with something inside
    valid_labels = list()

    while len(valid_labels) < 1:

        # Get a new top left_corner for the crop
        x = random.randint(0, d_w-CROP_W-1)
        y = random.randint(0, d_h-CROP_H-1)

        valid_labels = list()

        # Create variable for crop to compute overlap with labels
        bb1 = (x, x+CROP_W, y, y+CROP_H)

        for label in label_lines:
            # print(label)
            label = label.split()
            lx = int(label[1])
            ly = int(label[2])
            lw = int(label[3])
            lh = int(label[4])

            bb2 = (lx, lx+lw, ly, ly+lh)

            if get_intersection(bb1, bb2) > INTERSECTION_THRESHOLD:
                valid_labels.append(label)

    ### At this point I've generated a crop with at least a symbol inside
    
    # print("Synth diagram size: {}".format(synth_diagram.size))
    # print("x = {}, y = {}".format(x, y))

    # Crop the patch
    image_patch = synth_diagram.crop((x, y, x+CROP_W, y+CROP_H))
    
    patch_file_name = "{}.png".format(output_id)

    ### TODO to remove!
    # Write file to disk
    # with open(os.path.join(out_folder, patch_file_name), 'wb') as oi:
        # image_patch.save(oi, "PNG")

    # return

    with open(
            os.path.join(out_folder, "{}.txt".format(output_id)),
            'w'
    ) as lf:

        for label in valid_labels:
            class_label = label[0]
            lx = int(label[1])
            ly = int(label[2])
            lw = int(label[3])
            lh = int(label[4])

            # Recompute the coordinates of the label, adjusting for the new reference
            # system of the cropped patch
            lx_new = lx-x
            ly_new = ly-y

            # Bottom left corner
            lx2 = lx_new + lw
            ly2 = ly_new + lh

            # Now adjust the 4 coordinates
            # Top left corner
            lx_new = max(0, lx_new)
            ly_new = max(0, ly_new)

            # Bottom left corner
            lx2 = min(lx2, CROP_W)
            ly2 = min(ly2, CROP_H)

            # Now back to the x,y,w,h format
            lw_new = lx2-lx_new
            lh_new = ly2-ly_new
        
            # Write new label to file
            lf.write("{} {} {} {} {}\n".format(class_label, lx_new, ly_new, lw_new, lh_new))

    # Write file to disk
    with open(os.path.join(out_folder, patch_file_name), 'wb') as oi:
        image_patch.save(oi, "PNG")


def crop(args):

    IN_FOLDER = args.dataset_folder
    N_CROPS = args.n_crops
    OUT_FOLDER = args.output_folder

    if not os.path.exists(OUT_FOLDER):
        os.mkdir(OUT_FOLDER)

    for image_file in tqdm(os.listdir(IN_FOLDER)):
        # Only process png files
        if not image_file.endswith('.png'):
            continue

        image_id = image_file.split('.')[0]

        if args.verbose:
            print("Cropping file {}".format(image_id))

        # Load image
        synth_diagram = Image.open(
                os.path.join(IN_FOLDER, image_file)
        )

        # Load labels
        with open(
                os.path.join(IN_FOLDER, "{}.txt".format(image_id)),
                'r'
        ) as lf:
            label_lines = lf.readlines()

        for ci in range(N_CROPS):
            output_id = "{}_{}".format(image_id, ci)
            do_crop(synth_diagram.copy(), label_lines, output_id, OUT_FOLDER)

def main():

    args = parse_args()

    # Proceed to actual cropping
    crop(args)

if __name__ == '__main__':
    main()

