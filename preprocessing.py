import cv2
from bs4 import BeautifulSoup
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import csv
from utils import *

# from Cavity_detection.src.secrets import ROBOFLOW_API_KEY, ROBOFLOW_API_KEY_2, ROBOFLOW_API_KEY_3


"""## Directories and definitions"""
raw_data = 'raw'

data= 'preprocessing'
train_data = os.path.join(data, 'train')
interim_data = os.path.join(train_data, 'interim')
interim_caries = os.path.join(interim_data, 'caries')
interim_no_caries = os.path.join(interim_data, 'no_caries')

test_data = os.path.join(data, 'test')
test_interim_data = os.path.join(test_data, 'interim')
test_interim_caries = os.path.join(test_interim_data, 'caries')
test_interim_no_caries = os.path.join(test_interim_data, 'no_caries')
test_interim_caries_rois = os.path.join(test_interim_caries, 'rois')
test_rejected_rois = os.path.join(test_interim_caries, 'rejected_rois')

test_processed_data = os.path.join(test_data, 'processed')
test_processed_caries = os.path.join(test_processed_data, 'caries')
test_processed_no_caries = os.path.join(test_processed_data, 'no_caries')

interim_caries_rois = os.path.join(interim_caries, 'rois')
rejected_rois = os.path.join(interim_caries, 'rejected_rois')
processed_data = os.path.join(data, 'train', 'processed')
processed_caries = os.path.join(processed_data, 'caries')
processed_no_caries = os.path.join(processed_data, 'no_caries')

train_dataset = 'train_dataset'
test_dataset = 'test_dataset'

directories = [data, raw_data, train_data, interim_data, interim_caries, interim_no_caries,
               test_data, test_interim_data, test_interim_caries, test_interim_no_caries,
                test_interim_caries_rois, test_rejected_rois, test_processed_data,
                test_processed_caries, test_processed_no_caries, interim_caries_rois,
                rejected_rois, processed_data, processed_caries, processed_no_caries,
                train_dataset, test_dataset]


## cavity / no cavity separation

# Open each xml at raw_data and check for 'caries' in nametag to separate in folders
def split_classes(raw_data_path, train_path):
    interim_path = os.path.join(train_path, 'interim')
    interim_caries_path = os.path.join(interim_path, 'caries')
    interim_no_caries_path = os.path.join(interim_path, 'no_caries')

    # Create directories if they do not exist
    os.makedirs(interim_caries_path, exist_ok=True)
    os.makedirs(interim_no_caries_path, exist_ok=True)

    xmls = [xml for xml in os.listdir(raw_data_path) if xml.endswith('.xml')]
    for xml in xmls:
        with open(os.path.join(raw_data_path, xml)) as f:
            opened_xml = f.read()
        parsed_xml = BeautifulSoup(opened_xml, "xml") # reading xml
        filename = parsed_xml.find('filename').text
        if parsed_xml.find('name', string='caries') is not None: #check for caries nametag
            # copy xml and related image to cavities path
            shutil.copyfile(os.path.join(raw_data_path, xml), os.path.join(interim_caries_path, xml))
            shutil.copyfile(os.path.join(raw_data_path, filename), os.path.join(interim_caries_path, filename))
        else:
            # copy xml and related image to no cavities path
            shutil.copyfile(os.path.join(raw_data_path, xml), os.path.join(interim_no_caries_path, xml))
            shutil.copyfile(os.path.join(raw_data_path, filename), os.path.join(interim_no_caries_path, filename))



"""## Train/test split"""

def test_split(train_interim_path, test_path, split=0.1, seed=25):
    # Get paths for caries and no caries interim folders
    caries_path = os.path.join(train_interim_path, "caries")
    no_caries_path = os.path.join(train_interim_path, "no_caries")

    # Get list of JPG files for caries and no caries images
    caries_files = [f for f in os.listdir(caries_path) if f.endswith('.jpg')]
    no_caries_files = [f for f in os.listdir(no_caries_path) if f.endswith('.jpg')]

    # Calculate number of images based on split
    num_images = (len(caries_files) + len(no_caries_files))* split
    num_images = int(num_images)
    num_caries_images = num_images // 2
    num_no_caries_images = num_images - num_caries_images

    # Use seed value for random sample
    random.seed(seed)

    # Get random sample of caries and no caries images
    caries_files_to_move = random.sample(caries_files, num_caries_images)
    no_caries_files_to_move = random.sample(no_caries_files, num_no_caries_images)

    # Create test directories for caries and no caries images
    test_interim_path = os.path.join(test_path, "interim")
    test_caries_path = os.path.join(test_interim_path, "caries")
    test_no_caries_path = os.path.join(test_interim_path, "no_caries")

    os.makedirs(test_interim_path, exist_ok=True)
    os.makedirs(test_caries_path, exist_ok=True)
    os.makedirs(test_no_caries_path, exist_ok=True)

    # Move caries images and related XML files to test caries directory
    for f in caries_files_to_move:
        shutil.move(os.path.join(caries_path, f), os.path.join(test_caries_path, f))
        xml_file = f[:-3] + "xml"
        shutil.move(os.path.join(caries_path, xml_file), os.path.join(test_caries_path, xml_file))

    # Move no caries images and related XML files to test no caries directory
    for f in no_caries_files_to_move:
        shutil.move(os.path.join(no_caries_path, f), os.path.join(test_no_caries_path, f))
        xml_file = f[:-3] + "xml"
        shutil.move(os.path.join(no_caries_path, xml_file), os.path.join(test_no_caries_path, xml_file))


"""## caries roi creation"""

def extend_bounding_box(bbox, target_height, target_width, image_height, image_width):
    # bbox coordinates
    xmin, ymin, xmax, ymax = bbox
    bbox_height = ymax - ymin
    bbox_width = xmax - xmin

    # if bbox is bigger, update target values according to the original ratio (expand the smaller h or w)
    if bbox_height > target_height or bbox_width > target_width:
        ratio = target_height / target_width
        if bbox_height > bbox_width:
            target_height = bbox_height
            target_width = bbox_height / ratio
        if bbox_width > bbox_height:
            target_width = bbox_width
            target_height = bbox_width * ratio

    # pixels to extend
    height_delta = target_height - bbox_height
    width_delta = target_width - bbox_width

    # extend pixels while mantaining the bbox centered
    ymin = ymin - height_delta/2
    ymax = ymax + height_delta/2
    xmin = xmin - width_delta/2
    xmax = xmax + width_delta/2

    # extend to the other side if pixels were extender further than the original image borders
    if ymin < 0:
        ymax = ymax - ymin
        ymin = 0
    if ymax > image_width:
        height_delta = ymax - image_width
        ymax =  ymax - height_delta
        ymin = ymin - height_delta

    if xmin < 0:
        xmax = xmax - xmin
        xmin = 0
    if xmax > image_width:
        width_delta = xmax - image_width
        xmax = xmax - width_delta
        xmin = xmin - width_delta

    return [int(xmin), int(ymin), int(xmax), int(ymax)]

# Create and save roi's, applying clahe by default (needs to be applied to the full image first)
def roi_creation(images_path, save_dir, extend_bbox=False, apply_clahe:bool=True, visual_compare:bool=False, empty_roi_dir:bool=True, height:int=None, width:int=None, percentage:float=None, rejected_dir:str=None):
  if empty_roi_dir:
    empty_directory(save_dir)
    empty_directory(rejected_dir)

  cavity_counter = 0
  for xml in os.listdir(images_path):
      if xml.endswith('.xml'): # Check only xml files
        with open(file_path(images_path, xml)) as f:
          opened_xml = f.read()
        parsed_xml = BeautifulSoup(opened_xml, "xml")
        filename = parsed_xml.find('filename').text
        cavities = [cavity.find_parent() for cavity in parsed_xml.find_all('name', string='caries')] # List of cavities per file
        for cavity in cavities:
          # save coordinates of each cavity
          x_min = int(cavity.find('bndbox').find('xmin').string)
          x_max = int(cavity.find('bndbox').find('xmax').string)
          y_min = int(cavity.find('bndbox').find('ymin').string)
          y_max = int(cavity.find('bndbox').find('ymax').string)

          # verify roi ratio to image
          roi_height = y_max - y_min
          roi_width = x_max - x_min
          roi_area = roi_height * roi_width
          roi_ratio = roi_area / (height*width)

          img_filename = f'{cavity_counter}_{roi_ratio}.jpg'
          if roi_ratio < percentage:
            # save roi to rejected folder
            img = cv2.imread(file_path(images_path, filename),cv2.IMREAD_UNCHANGED) # read img as it is
            roi = img[ y_min:y_max,x_min:x_max] #axes are y:x
            cv2.imwrite(file_path(rejected_dir, img_filename),roi) # save roi to rejected_dir
          else:
            # processing and save roi to save_dir
            if extend_bbox:
              img = cv2.imread(file_path(images_path, filename),cv2.IMREAD_GRAYSCALE) # read img as grayscale
              img_width, img_height = img.shape
              bbox = [x_min, y_min, x_max, y_max]
              x_min, y_min, x_max, y_max = extend_bounding_box(bbox, height, width, img_height, img_width)
            if apply_clahe:
              img = cv2.imread(file_path(images_path, filename),cv2.IMREAD_GRAYSCALE) # read img as grayscale
              clahe_img = clahe.apply(img)
              roi = clahe_img[ y_min:y_max,x_min:x_max] #axes are y:x
            else:
              img = cv2.imread(file_path(images_path, filename),cv2.IMREAD_UNCHANGED) # read img as it is
              roi = img[ y_min:y_max,x_min:x_max] #axes are y:x

            cv2.imwrite(file_path(save_dir, img_filename),roi) # save roi to save_dir

          if visual_compare:
            # bounding_box for visualization
            boxA = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor='r', facecolor='none')
            # Image comparation
            fig, axes = plt.subplots(nrows=1, ncols=2)
            axes[0].imshow(img,cmap=plt.cm.gray)
            axes[0].add_patch(boxA)
            axes[1].imshow(roi,cmap=plt.cm.gray)
            fig.tight_layout()
            plt.rcParams['figure.figsize'] = [10, 10]
            plt.show()
            print(file_path(save_dir, str(cavity_counter) + '.jpg'), '\n')

          cavity_counter += 1


"""## no_caries roi creation"""

def extract_random_regions(src_dir, dest_dir, num_regions, height, width, apply_clahe=True):
    counter = 0
    filenames = [f for f in os.listdir(src_dir) if f.endswith(".jpg") or f.endswith(".png")]
    while counter < num_regions:
        filename = random.choice(filenames)
        img_path = os.path.join(src_dir, filename)

        # image reading and processing
        if apply_clahe:
          img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) # read img as grayscale
          clahe_img = clahe.apply(img)
        else:
          img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED) # read img as it is

        # extract region from apropiate
        if len(img.shape) == 2:
                h, w = img.shape
                c = 1
        else:
          h, w, c = img.shape

        # random coordinates of the region
        x1 = random.randint(0, w - width)
        y1 = random.randint(0, h - height)
        x2 = x1 + width
        y2 = y1 + height

        if c == 1:
          region = img[y1:y2, x1:x2]
        else:
          region = img[y1:y2, x1:x2, :]

        region_filename = f"{counter}.jpg"
        region_path = os.path.join(dest_dir, region_filename)
        cv2.imwrite(region_path, region)
        counter += 1


# Resize rois to be height x width
def process_images(input_dir, output_dir, height, width):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of files in input directory
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file in files:
        # Load image from file
        image = cv2.imread(os.path.join(input_dir, file), cv2.IMREAD_UNCHANGED)

        # Resize image
        result = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)

        # Save result to output directory
        cv2.imwrite(os.path.join(output_dir, file), result)


# Creates dataset in the required format for training and testing from the processed images
def create_dataset(src_dir, dest_dir, class_dict):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # Initialize a counter for the image filenames
    count = 0
    
    # Open the CSV file for writing
    with open(os.path.join(dest_dir, 'labels.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Loop through each class folder
        for class_name in class_dict.keys():
            class_dir = os.path.join(src_dir, class_name)
            
            # Loop through each image in the class folder
            for filename in os.listdir(class_dir):
                if filename.endswith('.jpg') or filename.endswith('.png'): # modify this line as per your image extensions
                    # Move the image to the destination directory and rename it
                    src_path = os.path.join(class_dir, filename)
                    dest_path = os.path.join(dest_dir, f'{count}.jpg') # modify extension here as per your image format
                    shutil.copy(src_path, dest_path)
                    
                    # Write the class identifier to the CSV file
                    writer.writerow([class_dict[class_name]])
                    
                    # Increment the counter
                    count += 1

if __name__ == "__main__":
    # clahe creation
    clahe = cv2.createCLAHE(clipLimit =10.0, tileGridSize=(8,8)) 

    # height and width
    height = 100
    width = 100

    # create and clean directories
    empty_directory(data)
    create_missing_folders(directories)

    # Preprocessing
    split_classes(raw_data, train_data)
    test_split(interim_data, test_data)

    # Train 
    roi_creation(interim_caries,interim_caries_rois, extend_bbox=True, visual_compare=False, height=height, width=width, percentage=0.5, rejected_dir=rejected_rois)
    caries_number = img_counter(interim_caries_rois)
    extract_random_regions(interim_no_caries,processed_no_caries,caries_number,height, width)
    process_images(interim_caries_rois, processed_caries,height, width)

    # Test
    roi_creation(test_interim_caries, test_interim_caries_rois, extend_bbox=True, visual_compare=False, height=height, width=width, percentage=0.5, rejected_dir=test_rejected_rois)
    caries_number = img_counter(test_interim_caries_rois)
    extract_random_regions(test_interim_no_caries, test_processed_no_caries,caries_number,height, width)
    process_images(test_interim_caries_rois, test_processed_caries,height, width)

    # Save train and test dataset

    class_dict = {'no_caries': 0, 'caries': 1}
    create_dataset(processed_data, train_dataset, class_dict)
    create_dataset(test_processed_data, test_dataset, class_dict)