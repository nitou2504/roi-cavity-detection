"""## Requirements """
import cv2
import os
import shutil
import matplotlib.pyplot as plt
import hashlib
import xml.etree.ElementTree as ET

"""## Useful functions"""

# Count imgs in a path
def img_counter(img_path:str):
  count = 0
  imgs = [img for img in os.listdir(img_path) if img.endswith('.jpg')]
  for img in imgs:
    count += 1
  # print('Total imgs:', count)
  return count

# Generate filepath
def file_path(base_path:str, filename:str, dir:bool=False):
  path = base_path+'/'+filename
  if dir:
    return path + '/'
  else:
     return path

# copy all files from one dir to another
def copy_all(source_folder, target_folder):
  files = os.listdir(source_folder)
  for file_name in files:
   shutil.copy(source_folder+'/'+file_name, target_folder+'/'+file_name)

# copy files and rename them as n+<i>.<extension>
def copy_all_rename(source_folder, target_folder, extension = '.jpg', start_number=0):
  files = os.listdir(source_folder)
  for i, file_name in enumerate(files):
   shutil.copy(source_folder+'/'+file_name, target_folder+'/'+f'{start_number+i}{extension}')

#searches for a specific subfolder in a base directory and then copies its contents to another folder:
def copy_subfolder_contents(base_directory, subfolder_name, target_directory):
    for root, dirs, files in os.walk(base_directory):
        if subfolder_name in dirs:
            source_directory = os.path.join(root, subfolder_name)
            for item in os.listdir(source_directory):
                s = os.path.join(source_directory, item)
                d = os.path.join(target_directory, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, False, None)
                else:
                    shutil.copy2(s, d)
    return

# delete all files from dir
def empty_directory_not_recursive(folder_path):
  files = os.listdir(folder_path)
  for file in files:
   os.remove(file_path(folder_path, file))

# delete all files and subdirectories
def empty_directory(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    return


def create_missing_folders(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # print(f"Created directory: {directory}")

def plot_img(img, img_path):
  image = cv2.imread(file_path(img_path, img),cv2.IMREAD_UNCHANGED) # read img as it is
  plt.imshow(image)



def rename_files_to_md5(folder_path):
    """
    Rename JPG and XML files in a folder to their MD5 hash, and update the filenames
    and paths inside the XML files to match the new names.

    Parameters:
    folder_path (str): The path to the folder containing the JPG and XML files.

    Returns:
    None.
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # Compute hash of the JPG file
            with open(os.path.join(folder_path, filename), 'rb') as f:
                hash = hashlib.md5(f.read()).hexdigest()

            # Rename JPG and XML files
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, hash + '.jpg'))
            os.rename(os.path.join(folder_path, os.path.splitext(filename)[0] + '.xml'), os.path.join(folder_path, hash + '.xml'))

            # Modify XML file to update filename and path
            xml_file = os.path.join(folder_path, hash + '.xml')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            root.find('filename').text = hash + '.jpg'
            root.find('path').text = hash + '.jpg'
            tree.write(xml_file)
