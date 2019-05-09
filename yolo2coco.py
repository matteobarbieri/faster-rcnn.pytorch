#!/usr/bin/env python
"""
Translates YOLO-style labels into coco-style json annotation.
Prints result to stdout (must be piped)
"""

import argparse

import os, sys
import datetime

import json

IMAGE_FILE_EXTENSION = 'png'

from tqdm import tqdm

def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1

def main():

    parser = argparse.ArgumentParser(
            description="Translates YOLO-style labels"
            "into coco-style json annotation."
    )
    
    # Path to the diagram image file
    parser.add_argument(
            'dataset_folder', 
            type=str, 
            help="Path to the dataset folder"
    )

    # Input image size
    parser.add_argument(
            '--image_w',
            type=int,
            help="Input image width",
            default=608
    )

    parser.add_argument(
            '--image_h',
            type=int,
            help="Input image height",
            default=608
    )

    parser.add_argument(
            '--min_area',
            type=int,
            help="Minimum area for an annotation to be included",
            default=100
    )

    parser.add_argument(
            '--names_file',
            type=str,
            help="Name of the file containing categories names",
            default="names.txt"
    )

    args = parser.parse_args()

    # IMAGE_W = IMAGE_H = 608
    IMAGE_W = args.image_w
    IMAGE_H = args.image_h

    YOLO_DATASET_FOLDER = args.dataset_folder

    # info{
            # "year" : int, 
            # "version" : str, 
            # "description" : str, 
            # "contributor" : str, 
            # "url" : str, 
            # "date_created" : datetime,
    # }

    ### Create the "info" part of the annotation
    info = {
            'year' : 2018,
            'version' : "0.1",
            'description' : "Synthetic dataset for ABB diagram understanding",
            'contributor' : "Matteo Barbieri",
            "url" : "nope",
            "date_created" : str(datetime.datetime.now())
    }

    ### Populate the 'categories' part of the annotation
    # names_file = os.path.join(YOLO_DATASET_FOLDER, args.names)
    # names_file = args.names_file

    # categories[{
            # "id" : int, 
            # "name" : str, 
            # "supercategory" : str,
    # }]

    categories_annotations = list()
    with open(args.names_file, 'r') as nf:

        cid = 1
        for nameline in nf.readlines():

            cann = {
                    "id" : cid, 
                    "name" : nameline.strip(), 
                    "supercategory" : "symbol",
            }

            cid += 1

            categories_annotations.append(cann)
        
    ### Now add actual annotations for images

    # First retrieve all the names of image files in the folder
    image_list = list()

    for f in os.listdir(YOLO_DATASET_FOLDER):
        if f.endswith(".{}".format(IMAGE_FILE_EXTENSION)):
            image_list.append(f)

    # Create empty lists for images and annotations
    image_annotations = list()
    labels_annotations = list()

    # Initialize the annotation id
    annotation_id = 1

    # Create the entries of the json file
    # for i, image in enumerate(image_list):
    for i, image in tqdm_enumerate(image_list):

        image_id = image[:-4]

        numeric_id = i+1

        # Create the entry for the image itself
        # image{
                # "id" : int, 
                # "width" : int, 
                # "height" : int, 
                # "file_name" : str, 
                # "license" : int, 
                # "flickr_url" : str, 
                # "coco_url" : str, 
                # "date_captured" : datetime,
        # }

        image_ann = {
                "id" : numeric_id, 
                "width" : IMAGE_W, 
                "height" : IMAGE_H, 
                "file_name" : image, 
                "license" : 1, 
                "flickr_url" : "", 
                "coco_url" : "", 
                "date_captured" : str(datetime.datetime.now()),
        }

        # Add to the image list
        image_annotations.append(image_ann)

        with open(os.path.join(YOLO_DATASET_FOLDER, "{}.txt".format(image_id))) as yaf:
            yolo_lines = yaf.readlines()

            for yolo_line in yolo_lines:

                # For each line create an annotation
                # print(yolo_line.strip())
                yolo_larr = yolo_line.strip().split()
                # print(yolo_larr)

                # Extract category ID and bounding box coordinates from array
                category_id = int(yolo_larr[0])
                # category_id = int(yolo_larr[0]) + 1

                # # TODO workaround for blabla
                # if category_id == 90:
                    # category_id = 89
                
                x = int(yolo_larr[1])
                y = int(yolo_larr[2])
                w = int(yolo_larr[3])
                h = int(yolo_larr[4])

                if w*h < args.min_area:
                    # print("catid: {}, w: {}, h: {}".format(category_id, w, h))
                    # break
                    continue

                # annotation{
                        # "id" : int, 
                        # "image_id" : int, 
                        # "category_id" : int, 
                        # "segmentation" : RLE or [polygon], 
                        # "area" : float, 
                        # "bbox" : [x,y,width,height], 
                        # "iscrowd" : 0 or 1,
                # }

                label_annotation = {
                        "id" : annotation_id, 
                        "image_id" : numeric_id, 
                        "category_id" : category_id, 
                        "segmentation" : [],
                        "area" : w*h, 
                        "bbox" : [x,y,w,h], 
                        "iscrowd" : 0, # fixed to 0 for this specific problem
                }

                # Increment the counter for annotations ID
                annotation_id += 1

                labels_annotations.append(label_annotation)

    annotation_dict = {
            'info' : info,
            'images' : image_annotations,
            'annotations' : labels_annotations,
            'categories' : categories_annotations
    }

    json_string = json.dumps(annotation_dict)
    print(json_string)

if __name__ == '__main__':
    main()
