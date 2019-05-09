# Training instructions

This document contains the instructions for how to train a model for a new
notation.

## Dataset preparation

### Cropping

Original diagram images are too big to be processed in one go, therefore it is
necessary to generate several crops of the image. The included script
`generate_training_crops.py` performs a given number of fixed size random crops
for each image (20 by default), adjusting labels as required.

This script assumes that the labels in the original images are in the YOLO 
format (see https://github.com/AlexeyAB/Yolo_mark/issues/60).

### Labels

The algorithm for training requires that the dataset annotations are in the
_COCO_ format (see
http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch for
more info on the annotation format).

In case the images have been annotated using the _YOLO_ format, it is possible 
to use the included script `yolo2coco.py` to convert the annotations in the 
_COCO_ format.

### Dataset class files

It is necessary to create a file named `<NOTATION>.py` in folder `lib/datasets`,
defining some properties of the notation dataset. Copy the file related to an
existing notation and apply changes where needed.

Moreover, it is necessary to add the required entries in the following files
(search the code for string `abbdoc` to see examples on previous datasets):

 * `lib/datasets/factory.py`
 * `lib/roi_data_layer/roidb.py`

Dataset folders must be placed in a specific folder (see file `trainval_net.py`,
around line 184).

## Training

To launch the training, simply use the script `do_training.sh`, passing notation
name and model destination as arguments.

## Model deployment

Once the model has been trained, it is necessary to perform a few steps in order
to add it to the list of the ones available for inference.

 1. Go on the Azure storage page, select `bim2bomstorage` (BLOBs).
 1. Select the `inference-models` container.
 1. Navigate to the chosen version of models (currently `v2`).
 1. Upload `three` files in a subfolder named as the notation (lowercase,
    removing all spaces):
    1. The `.pth` file of the model.
    1. A text file containing the names of the categories.
    1. A text file containing the mapping from the category ids defined in the 
       metamodel file/db to the id used for inference.
 1. Edit the `notations-list.txt` file, adding the name of the added notation
    (lowercase, removing all spaces).
