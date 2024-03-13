#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2024/2/19 15:45
# @Author: GuuX
# @File  : Preprocessing.py

import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import random
import csv
import shutil
import json

from PIL import Image
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from collections import defaultdict
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm


def crop_images(ori_path, crop_path, crop_size, step_size, keepname=False, sample_rate=1.0):
    """
    Crop images in a directory with a sliding window approach and save the cropped images to a new directory.

    :param ori_path: Path to the original images.
    :param crop_path: Path where cropped images will be saved.
    :param crop_size: Size of the crop (width and height in pixels).
    :param step_size: Step size for the sliding window (in pixels).
    :param keepname: Whether to keep the original filename in the cropped image filename.
    :param sample_rate: Fraction of images to randomly sample and crop from the original directory.
    """
    window_counts = []
    filenames = [f for f in os.listdir(ori_path) if f.lower().endswith('.jpg')]
    sampled_indices = random.sample(range(len(filenames)), int(len(filenames) * sample_rate))

    print(f"Cropping from {ori_path} to {crop_path}, \n"
          f"crop size is {crop_size}, step size is {step_size}, and sampling rate set to {sample_rate}")

    progress = Progress(SpinnerColumn(),
                        *Progress.get_default_columns(),
                        TimeElapsedColumn(), )
    with progress:
        task_id = progress.add_task(f"[cyan]Sliding window cropping total [red]{len(sampled_indices)} [cyan]images: ",
                                    total=len(sampled_indices))
        for i in sampled_indices:
            filename = filenames[i]

            image = cv2.imread(os.path.join(ori_path, filename))
            windows = sliding_window(image, step_size, crop_size)
            window_num = len(windows)

            for j, window in enumerate(windows):
                x1, x2, y1, y2 = window
                crop = image[y1:y2, x1:x2]
                if keepname:
                    cv2.imwrite(os.path.join(crop_path, f'{filename[:-4]}_{i}_{j + 1}.jpg'), crop)
                else:
                    cv2.imwrite(os.path.join(crop_path, f'crop_{i}_{j + 1}.jpg'), crop)

            window_counts.append(window_num)
            progress.update(task_id, advance=1, refresh=True)
    return window_counts


def sliding_window(image, step_size, window_size):
    """
    Generate a list of coordinates for cropping the image using the sliding window method.

    :param image: The image to crop.
    :param step_size: The step size for moving the window.
    :param window_size: The size of the window.
    :return: A list of tuples containing the coordinates for cropping.
    """
    range_list = []
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            xmin = x
            xmax = min(x + window_size, image.shape[1])
            ymin = y
            ymax = min(y + window_size, image.shape[0])
            range_list.append((xmin, xmax, ymin, ymax))
    return range_list


def parse_obj(xml_path, filename):
    """
    Parse an XML file to extract object information.

    :param xml_path: Path to the directory containing the XML files.
    :param filename: The XML file name.
    :return: A list of dictionaries, each containing information about an object.
    """
    tmp_path = os.path.join(xml_path, filename)
    tree = ET.parse(tmp_path)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {'name': obj.find('name').text}
        objects.append(obj_struct)
    return objects


def class_count_xml(xml_path, savetxt=False, savecsv=False, savepath=None):
    """
    Count the occurrences of each class in XML files and optionally save the results to a text or CSV file.

    :param xml_path: Path to the directory containing the XML files.
    :param savetxt: Whether to save the results to a text file.
    :param savecsv: Whether to save the results to a CSV file.
    :param savepath: Path where the result file will be saved.
    """
    if (savetxt or savecsv) and savepath is None:
        print('Please set the savepath of the result file!\n'
              'The result will not be saved this time.')
        return

    filenames = [name for name in os.listdir(xml_path) if name.endswith('.xml')]
    num_objs = {}
    classnames = []

    for name in filenames:
        objects = parse_obj(xml_path, name)
        for obj in objects:
            obj_name = obj['name']
            num_objs[obj_name] = num_objs.get(obj_name, 0) + 1
            if obj_name not in classnames:
                classnames.append(obj_name)

    if savetxt or savecsv:
        with open(os.path.join(savepath, f'class_count.{"txt" if savetxt else "csv"}'), 'w', newline='') as file:
            if savetxt:
                file.write('spe,num\n')
            else:
                writer = csv.writer(file)
                writer.writerow(['spe', 'num'])
            for name in classnames:
                print(f'{name}:{num_objs[name]}')
                if savetxt:
                    file.write(f'{name},{num_objs[name]}\n')
                else:
                    writer.writerow([name, num_objs[name]])
        print(f'Result file saved to {savepath}!')
    else:
        for name in classnames:
            print(f'{name}:{num_objs[name]} ind')

    print('Finish counting!')


def class_count_txt(txt_dir, classes):
    """
    Count occurrences of each class in text files within a given directory.

    :param txt_dir: Directory containing text files.
    :param classes: List of class names.
    """
    class_counts = {c: 0 for c in classes}

    for txt_file in os.listdir(txt_dir):
        if txt_file.endswith('.txt'):
            txt_path = os.path.join(txt_dir, txt_file)

            # Skip empty files
            if os.stat(txt_path).st_size == 0:
                continue

            with open(txt_path) as f:
                for line in f:
                    if line.strip():  # Check if line is not empty
                        class_id = int(line.split()[0])
                        class_counts[classes[class_id]] += 1

    for class_name, count in class_counts.items():
        print(f'{class_name}: {count}')


def rename_objects_in_xml_one(path, savepath, old_name, new_name):
    """
    Rename a specified object in XML files if the old name exactly matches.

    :param path: Directory containing XML files.
    :param savepath: Directory where modified XML files are saved.
    :param old_name: The old object name to be replaced.
    :param new_name: The new object name.
    """
    for file in os.listdir(path):
        if file.endswith(".xml"):
            in_file = os.path.join(path, file)
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                if obj.find('name').text == old_name:
                    obj.find('name').text = new_name

            new_savepath = os.path.join(savepath, file)
            tree.write(new_savepath)


def rename_objects_in_xml_more(path, savepath, old_names, new_name):
    """
    Rename multiple objects within XML files based on a list of old names.

    :param path: Directory containing XML files.
    :param savepath: Directory where modified XML files are saved.
    :param old_names: List of old names to be replaced.
    :param new_name: The new object name.
    """
    for file in os.listdir(path):
        if file.endswith(".xml"):
            in_file = os.path.join(path, file)
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.findall('object'):
                if obj.find('name').text in old_names:
                    obj.find('name').text = new_name

            new_savepath = os.path.join(savepath, file)
            tree.write(new_savepath)


def splitratio(txt_path, pic_path, dataset_path, ratio=0.9):
    """
    Splits the dataset based on the given ratio, copying images and labels to respective directories.

    :param txt_path: Directory of text files.
    :param pic_path: Directory of picture files.
    :param dataset_path: Base directory for the split dataset.
    :param ratio: Ratio of training data to total data.
    """

    # Helper functions for copying images and text files
    def copy_images(set_type, image_id):
        in_file = os.path.join(pic_path, f'{image_id}.jpg')
        out_file = os.path.join(dataset_path, f'images/{set_type}/{image_id}.jpg')
        shutil.copy(in_file, out_file)

    def copy_txt(set_type, txt_id):
        in_file = os.path.join(txt_path, f'{txt_id}.txt')
        out_file = os.path.join(dataset_path, f'labels/{set_type}/{txt_id}.txt')
        shutil.copy(in_file, out_file)

    # Ensure directories exist
    for sub_path in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dataset_path, sub_path), exist_ok=True)

    total_txt = os.listdir(txt_path)
    num = len(total_txt)
    tv = int(num * ratio)
    train = random.sample(range(num), tv)

    with open(os.path.join(dataset_path, 'train.txt'), 'w') as ftrain, \
            open(os.path.join(dataset_path, 'val.txt'), 'w') as fval:

        for i, txt_file in enumerate(total_txt):
            name = txt_file[:-4]
            set_type = 'train' if i in train else 'val'
            txt = f"{dataset_path}/images/{set_type}/{name}.jpg\n"
            copy_images(set_type, name)
            copy_txt(set_type, name)
            if set_type == 'train':
                ftrain.write(txt)
            else:
                fval.write(txt)

    print('Dataset has been split and set into the document frame!')


def read_xml_annotation(xml_file):
    """
    Reads a single XML file and returns all object bounding boxes.

    :param xml_file: Path to the XML file.
    :return: List of tuples representing bounding boxes (xmin, ymin, xmax, ymax).
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def get_boxes_from_xmls(xml_dir):
    """
    Iterates over all XML files in a given directory, collecting all bounding box widths and heights.

    :param xml_dir: Directory containing XML files.
    :return: Numpy arrays of widths and heights of all bounding boxes.
    """
    widths, heights = [], []

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            boxes = read_xml_annotation(os.path.join(xml_dir, file))
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                widths.append(xmax - xmin)
                heights.append(ymax - ymin)

    return np.array(widths), np.array(heights)


def plot_distribution_and_calculate_statistics(widths, heights):
    """
    Plots the distribution of bounding box widths and heights, and calculates statistical data.

    :param widths: Numpy array of bounding box widths.
    :param heights: Numpy array of bounding box heights.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(widths, kde=True)
    plt.title('Width Distribution')
    plt.subplot(1, 2, 2)
    sns.histplot(heights, kde=True)
    plt.title('Height Distribution')
    plt.show()

    print(f'Widths: mean = {np.mean(widths)}, median = {np.median(widths)}')
    print(f'Heights: mean = {np.mean(heights)}, median = {np.median(heights)}')


def perform_kmeans_clustering(widths, heights, n_clusters=9):
    """
    Performs K-Means clustering to find the optimal anchor box sizes.

    :param widths: Numpy array of bounding box widths.
    :param heights: Numpy array of bounding box heights.
    :param n_clusters: Number of clusters to form.
    :return: List containing the anchor box sizes for each cluster.
    """
    sizes = np.stack([widths, heights], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(sizes)
    anchors = sorted(kmeans.cluster_centers_, key=lambda x: x[0] * x[1])

    return anchors


def anchor_kmeans_plusplus(xml_dir, output_dir):
    """
    Main function to process XML files, plot distributions, calculate statistics, and determine anchor boxes.

    :param xml_dir: Directory containing XML files.
    :param output_dir: Directory to save output files.
    """
    widths, heights = get_boxes_from_xmls(xml_dir)
    np.savetxt(os.path.join(output_dir, "bbox_widths.csv"), widths, delimiter=",")
    np.savetxt(os.path.join(output_dir, "bbox_heights.csv"), heights, delimiter=",")

    plot_distribution_and_calculate_statistics(widths, heights)
    anchors = perform_kmeans_clustering(widths, heights)

    for i, layer in enumerate(anchors):
        print(f'Layer {i}: {layer}')


def rotate_and_check_image_size(image_path, desired_width=6000, desired_height=4000):
    """
    Rotates an image if necessary and checks if it matches the desired dimensions.

    :param image_path: Path to the image.
    :param desired_width: Desired width of the image.
    :param desired_height: Desired height of the image.
    """
    try:
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        if width < height:
            # Rotate the image 90 degrees counterclockwise
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        height, width, _ = image.shape

        if width == desired_width and height == desired_height:
            cv2.imwrite(image_path, image)
        else:
            print(f"Skipping {image_path}: The image does not have the desired size.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def flip_images(folder_path, desired_width=6000, desired_height=4000):
    """
    Flips images in a folder to match the desired dimensions.

    :param folder_path: Path to the folder containing images.
    :param desired_width: Desired width of the images.
    :param desired_height: Desired height of the images.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            rotate_and_check_image_size(image_path, desired_width, desired_height)


def balance_dataset(img_dir, txt_dir, out_img_dir, out_txt_dir, target_classes, class_counts_path, classes):
    """
    Copies images and their annotation files to new directories to balance the dataset according to specified class occurrence conditions.

    Parameters:
    - img_dir: Source directory containing images.
    - txt_dir: Source directory containing corresponding annotation text files.
    - out_img_dir: Target directory for copied images.
    - out_txt_dir: Target directory for copied annotation text files.
    - target_classes: List of class IDs to be considered for balancing.
    - class_counts_path: Path to the text file containing initial class counts.
    """
    # classes = ['Tubuca arcuata', 'Austruca lactea', 'Gelasimus spp', 'Paraleptuca splendida', 'Austruca annulipes',
    #            'Tubuca paradussumieri',
    #            'Parasesarma affine', 'Parasesarma eumolpe', 'Parasesarma continentale', 'Sesarmidae spp',
    #            'Macrophthalmus tomentosus', 'Macrophthalmus pacificus',
    #            'Metaplax spp', 'Metaplax elegans',
    #            'Ilyoplax formosensis', 'Ilyoplax serrata', 'Tmethypocoelis ceratophora',
    #            'SWL', 'Paracleistostoma depressum', 'Scopimera intermedia']

    # Ensure target directories exist
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_txt_dir, exist_ok=True)

    # Read original class counts
    original_counts = {}
    class_name_to_id = {name: i for i, name in enumerate(classes)}
    with open(class_counts_path) as f:
        for line in f:
            class_name, count = line.strip().split(',')
            if class_name in class_name_to_id:  # Ensure class name is in the predefined list
                cid = class_name_to_id[class_name]
                original_counts[cid] = int(count)

    print('Original counts:', original_counts)

    class_counts = defaultdict(int)

    max_num = 10000
    min_num = 500

    file_list = os.listdir(img_dir)
    random.shuffle(file_list)
    print_interval = 50

    for i, filename in enumerate(file_list):
        if not filename.endswith('.jpg'):
            continue

        img_path = os.path.join(img_dir, filename)
        txt_path = os.path.join(txt_dir, filename.replace('.jpg', '.txt'))
        out_img_path = os.path.join(out_img_dir, filename.split('.')[0] + '_add.' + filename.split('.')[1])
        out_txt_path = os.path.join(out_txt_dir, filename.replace('.jpg', '_add.txt'))

        if not os.path.exists(txt_path):
            continue

        with open(txt_path) as f:
            lines = f.readlines()

        class_ids = [int(line.split()[0]) for line in lines]

        copy_img = False

        for cid in class_ids:
            if original_counts.get(cid, 0) >= min_num and original_counts.get(cid, 0) + class_counts[cid] < max_num:
                copy_img = True
                break

        if copy_img:
            shutil.copyfile(img_path, out_img_path)
            shutil.copyfile(txt_path, out_txt_path)

            for cid in class_ids:
                class_counts[cid] += 1

        if i % print_interval == 0 and i > 0:
            print(f'Processed {i} images. Current class counts:')
            for cid in target_classes:
                print(f'{cid}: {class_counts[cid]} images')


def read_classes_from_file(filepath):
    """
    Reads class names from a given text file, with one class per line.

    Parameters:
    - filepath: Path to the text file containing class names.

    Returns:
    - A list of class names.
    """
    with open(filepath, 'r') as file:
        classes = [line.strip() for line in file.readlines()]
    return classes


def crop_objects_from_images(img_path, xml_path, obj_img_path, classes_file):
    """
    Crops objects from images based on annotations in corresponding XML files and saves them into specified directories.

    Parameters:
    - img_path: Source directory containing images.
    - xml_path: Source directory containing XML annotation files.
    - obj_img_path: Target directory for saving cropped images.
    - classes_file: Path to the text file containing class names.
    """
    classes = read_classes_from_file(classes_file)  # Dynamically load class names from a file

    for img_file in os.listdir(img_path):
        if img_file[-4:] in ['.bmp', '.jpg', '.png']:
            img_filename = os.path.join(img_path, img_file)
            img_cv = cv2.imread(img_filename)

            img_name = os.path.splitext(img_file)[0]
            xml_name = os.path.join(xml_path, f'{img_name}.xml')

            if os.path.exists(xml_name):
                root = ET.parse(xml_name).getroot()
                count = 0

                for obj in root.iter('object'):
                    name = obj.find('name').text
                    if name not in classes:
                        continue
                    cls_id = classes.index(name)

                    xmlbox = obj.find('bndbox')
                    x0, y0 = xmlbox.find('xmin').text, xmlbox.find('ymin').text
                    x1, y1 = xmlbox.find('xmax').text, xmlbox.find('ymax').text

                    obj_img = img_cv[int(float(y0)):int(float(y1)), int(float(x0)):int(float(x1))]

                    class_doc = os.path.join(obj_img_path, str(cls_id))
                    os.makedirs(class_doc, exist_ok=True)

                    cv2.imwrite(os.path.join(class_doc, f'{img_name}_{count}.jpg'), obj_img)
                    count += 1

    print(f"Objects have been cropped based on XML files and saved at {obj_img_path}!")


def create_map(cate, map_txt_path):
    """
    Creates a mapping file for class names to class IDs.

    :param cate: A list of class names.
    :param map_txt_path: The path where the class mapping file will be saved.
    """
    # Create a dictionary mapping class IDs to class names
    dict_map = dict(enumerate(cate))
    # Convert the dictionary to a JSON string
    json_map = json.dumps(dict_map, ensure_ascii=False)
    # Write the JSON string to the specified file
    with open(map_txt_path, 'w', encoding='utf-8') as f:
        f.write(json_map)

    print('Class map created:', json_map)


def crab_map_make(ori_folder, train_file_save_path, classes, map_txt_path):
    """
    Generates training file mappings and creates a class mapping file.

    :param ori_folder: The path to the original folder containing class folders.
    :param train_file_save_path: The path where the training files should be saved.
    :param classes: A list of class names.
    :param map_txt_path: The path where the class mapping file will be saved.
    """
    # List all class folders in the original folder
    cls_item = os.listdir(ori_folder)

    # Construct paths for the training text and CSV files
    txt_path = os.path.join(train_file_save_path, 'EFN-train.txt')
    csv_path = os.path.join(train_file_save_path, 'EFN-train.csv')

    # Using 'with' statement to ensure files are properly closed after their blocks are left
    with open(txt_path, 'w', encoding='utf-8') as train_file, open(csv_path, 'w', newline='',
                                                                   encoding='utf-8') as csvfile:
        # Initialize CSV writer
        train_csv = csv.writer(csvfile)
        # Write column headers
        train_file.write('FileID SpeciesID\n')
        train_csv.writerow(['FileID', 'SpeciesID'])

        # Iterate through each class folder
        for cls in cls_item:
            # List all images within the class folder
            images = os.listdir(os.path.join(ori_folder, cls))

            # Iterate through each image
            for tp in images:
                # Extract image ID (filename without extension)
                images_id = os.path.splitext(tp)[0]
                # Write to both text and CSV files
                train_file.write(f'{images_id} {cls}\n')
                train_csv.writerow([images_id, cls])

    # After files are written, create the class mapping file
    create_map(classes, map_txt_path)

    print('Mapping creation finished.')


class ImageProcessor:
    def __init__(self, input_txt_folder, input_image_folder, output_image_folder):
        self.input_txt_folder = input_txt_folder
        self.input_image_folder = input_image_folder
        self.output_image_folder = output_image_folder
        self.ensure_output_folder_exists()

    def ensure_output_folder_exists(self):
        if not os.path.exists(self.output_image_folder):
            os.makedirs(self.output_image_folder)

    def process_images(self):
        txt_files = [f for f in os.listdir(self.input_txt_folder) if f.endswith('.txt')]
        for txt_file in txt_files:
            self.process_single_image(txt_file)
        print("Batch processing completed.")

    def process_single_image(self, txt_file):
        image_filename = os.path.splitext(txt_file)[0] + '.jpg'
        image_path = os.path.join(self.input_image_folder, image_filename)
        if not os.path.exists(image_path):
            print(f"Image file {image_path} does not exist. Skipping...")
            return

        image = cv2.imread(image_path)
        imgsz = image.shape
        image = self.undistort_image(image, imgsz)

        rectangles = self.read_rectangles_from_txt(txt_file, imgsz)
        if len(rectangles) < 4:
            print(f"Insufficient rectangles in {txt_file}. Skipping...")
            return

        in_rects = self.intersection(rectangles, imgsz)
        internal_vertices = self.sort_rect(in_rects)

        result = self.apply_perspective_transform(image, internal_vertices)
        self.save_transformed_image(txt_file, result)

    def undistort_image(self, image, imgsz):
        camera_matrix = np.array([[4.26099396e+03, 0, 2.98271686e+03], [0, 4.25877240e+03, 1.97573091e+03], [0, 0, 1]])
        dist_coeffs = np.array([-1.66168524e-01, 2.72803262e-01, 6.75952909e-04, -1.89324745e-04, -2.48393477e-01])
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, imgsz[:2][::-1], 0)
        return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    def read_rectangles_from_txt(self, txt_file, imgsz):
        txt_file_path = os.path.join(self.input_txt_folder, txt_file)
        rectangles = []
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                cls, x, y, w, h = map(float, line.split())
                rectangles.append(self.calculate_vertices(x, y, w, h, imgsz))
        return rectangles

    def calculate_vertices(self, x, y, w, h, imgsz):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y - h / 2
        x3, y3 = x - w / 2, y + h / 2
        x4, y4 = x + w / 2, y + h / 2
        return [(x1 * imgsz[1], y1 * imgsz[0]), (x2 * imgsz[1], y2 * imgsz[0]), (x3 * imgsz[1], y3 * imgsz[0]),
                (x4 * imgsz[1], y4 * imgsz[0])]

    def intersection(self, rectangles, imgsz):
        image_center = (imgsz[1] / 2, imgsz[0] / 2)
        in_rect = []
        for rect in rectangles:
            min_dist = float('Inf')
            closest_vertex = None
            for vertex in rect:
                x, y = vertex
                dist = np.sqrt((x - image_center[0]) ** 2 + (y - image_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_vertex = (x, y)
            in_rect.append(closest_vertex)
        return in_rect

    def sort_rect(self, in_rect):
        points = np.array(in_rect)
        left_top = min(points, key=lambda p: p[0] + p[1])
        right_top = min(points, key=lambda p: -p[0] + p[1])
        left_bottom = min(points, key=lambda p: p[0] - p[1])
        right_bottom = min(points, key=lambda p: -p[0] - p[1])
        return [left_top, right_top, left_bottom, right_bottom]

    def apply_perspective_transform(self, image, internal_vertices):
        target_size = 2560
        target_vertices = [(0, 0), (target_size, 0), (0, target_size), (target_size, target_size)]
        matrix = cv2.getPerspectiveTransform(np.array(internal_vertices, dtype=np.float32),
                                             np.array(target_vertices, dtype=np.float32))
        return cv2.warpPerspective(image, matrix, (target_size, target_size))

    def save_transformed_image(self, txt_file, result):
        output_image_filename = os.path.splitext(txt_file)[0] + '.jpg'
        output_image_path = os.path.join(self.output_image_folder, output_image_filename)
        cv2.imwrite(output_image_path, result)


# # Usage sample
# input_txt_folder = 'D:/MEE-EXP/LuoYuan-exp/txt'
# input_image_folder = 'D:/DataSource/LuoYuan/101___05'
# output_image_folder = 'D:/MEE-EXP/LuoYuan-exp/crop'
# processor = ImageProcessor(input_txt_folder, input_image_folder, output_image_folder)
# processor.process_images()

class TxtToXmlConverter:
    def __init__(self, input_dir, output_dir, img_width, img_height, categ_path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_width = img_width
        self.img_height = img_height
        self.categ_path = categ_path
        self.classes = self.load_categories()

    def load_categories(self):
        # Load category names from a file
        classes = []
        with open(self.categ_path, 'r') as file:
            for line in file:
                classes.append(line.strip())
        print(classes)
        return classes

    @staticmethod
    def convert(size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = box[0] / dw
        y = box[1] / dh
        w = box[2] / dw
        h = box[3] / dh
        return [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)]

    def txt_to_xml(self, txt_file, xml_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        node_root = Element('annotation')

        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'crabcomm'

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = os.path.basename(txt_file).replace('.txt', '.jpg')

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(self.img_width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(self.img_height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        for line in lines:
            obj = line.strip().split()

            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = self.classes[int(obj[0])]
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')

            b = (float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]))
            bb = self.convert((self.img_width, self.img_height), b)
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_ymax = SubElement(node_bndbox, 'ymax')

            node_xmin.text = str(bb[0])
            node_ymin.text = str(bb[1])
            node_xmax.text = str(bb[2])
            node_ymax.text = str(bb[3])

        xml = tostring(node_root, 'utf-8')
        dom = parseString(xml)

        with open(xml_file, 'wb') as f:
            f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

    def convert_all_txt_to_xml(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        txt_files = [f for f in os.listdir(self.input_dir) if f.endswith('.txt')]
        for txt_file in txt_files:
            txt_path = os.path.join(self.input_dir, txt_file)
            xml_path = os.path.join(self.output_dir, txt_file.replace('.txt', '.xml'))
            self.txt_to_xml(txt_path, xml_path)
        print('Conversion finished!')


# # Usage sample
# input_dir = 'path/to/your/input/dir'
# output_dir = 'path/to/your/output/dir'
# img_width = 1920
# img_height = 1080
# classes = ['Tubuca arcuata', 'Austruca lactea', 'Gelasimus spp', 'Paraleptuca splendida',
#                         'Austruca annulipes', 'Tubuca paradussumieri',
#                         'Parasesarma affine', 'Parasesarma eumolpe', 'Parasesarma continentale', 'Sesarmidae spp',
#                         'Macrophthalmus tomentosus', 'Macrophthalmus pacificus',
#                         'Metaplax spp', 'Metaplax elegans',
#                         'Ilyoplax formosensis', 'Ilyoplax serrata', 'Tmethypocoelis ceratophora',
#                         'SWL', 'Paracleistostoma depressum', 'Scopimera intermedia']
# converter = TxtToXmlConverter(input_dir, output_dir, img_width, img_height)
# converter.convert_all_txt_to_xml()

class XMLToTxtConverter:
    def __init__(self, categ_path, txt_path, xml_path, img_path):
        # Initialize the converter with paths to categories, txt, xml, and image directories
        self.categ_path = categ_path
        self.txt_path = txt_path
        self.xml_path = xml_path
        self.img_path = img_path
        # Load classes from a categories file or directly define them here
        self.classes = self.load_categories()

    def load_categories(self):
        # Load category names from a file
        classes = []
        with open(self.categ_path, 'r') as file:
            for line in file:
                classes.append(line.strip())
        return classes

    def convert_box_size(self, size, box):
        # Convert XML box size to YOLO format
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        return (x * dw, y * dh, w * dw, h * dh)

    def convert_annotation(self, image_id):
        # Convert a single XML file to a TXT annotation file
        txt_file_path = os.path.join(self.txt_path, f'{image_id}.txt')
        xml_file_path = os.path.join(self.xml_path, f'{image_id}.xml')

        with open(txt_file_path, 'w') as out_file:
            if os.path.exists(xml_file_path):
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in self.classes or int(difficult) == 1:
                        continue
                    cls_id = self.classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = self.convert_box_size((w, h), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    def convert_all(self):
        # Convert all XML annotations in the directory to TXT format
        if not os.path.exists(self.txt_path):
            os.makedirs(self.txt_path)
        for image_file in os.listdir(self.img_path):
            image_id = os.path.splitext(image_file)[0]
            self.convert_annotation(image_id)
        print('All XML files have been converted to TXT format.')


# # Usage sample
# categ_path = "D:/MEE-EXP/categories.txt"
# xml_path = 'D:/MEE-EXP/crop_test/xml_man/'
# txt_path = 'D:/MEE-EXP/crop_test/crab-man/'
# img_path = 'D:/MEE-EXP/crop_test/crop'
# converter = XMLToTxtConverter(categ_path, txt_path, xml_path, img_path)
# converter.convert_all()
import collections
import datetime
import glob
import sys
import uuid
import imgviz
import labelme
import os.path as osp
import pycocotools.mask

# Revised and organized from https://github.com/labelmeai/labelme/blob/12d425b956878132566d243b4d9f6f3af33ec810/examples/bbox_detection/labelme2voc.py#L70.
# Wada, K. Labelme: Image Polygonal Annotation with Python [Computer software]. https://doi.org/10.5281/zenodo.5711226
class LabelmeToJSON:
    def __init__(self, input_dir, output_dir, labels_file, noviz=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.labels_file = labels_file
        self.noviz = noviz

    def convert(self):
        if osp.exists(self.output_dir):
            print("Output directory already exists:", self.output_dir)
            sys.exit(1)
        os.makedirs(self.output_dir)
        os.makedirs(osp.join(self.output_dir, "JPEGImages"))
        if not self.noviz:
            os.makedirs(osp.join(self.output_dir, "Visualization"))
        print("Creating dataset:", self.output_dir)

        now = datetime.datetime.now()

        data = dict(
            info=dict(
                description=None,
                url=None,
                version=None,
                year=now.year,
                contributor=None,
                date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            ),
            licenses=[
                dict(
                    url=None,
                    id=0,
                    name=None,
                )
            ],
            images=[
                # license, url, file_name, height, width, date_captured, id
            ],
            type="instances",
            annotations=[
                # segmentation, area, iscrowd, image_id, bbox, category_id, id
            ],
            categories=[
                # supercategory, id, name
            ],
        )

        class_name_to_id = {}
        for i, line in enumerate(open(self.labels_file).readlines()):
            class_id = i  # starts with 0 changed by [GuuX] this project
            class_name = line.strip()
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            class_name_to_id[class_name] = class_id
            data["categories"].append(
                dict(
                    supercategory=None,
                    id=class_id,
                    name=class_name,
                )
            )

        out_ann_file = osp.join(self.output_dir, "annotations.json")
        label_files = glob.glob(osp.join(self.input_dir, "*.json"))
        for image_id, filename in enumerate(label_files):
            print("Generating dataset from:", filename)

            label_file = labelme.LabelFile(filename=filename)

            base = osp.splitext(osp.basename(filename))[0]
            out_img_file = osp.join(self.output_dir, "JPEGImages", base + ".jpg")

            img = labelme.utils.img_data_to_arr(label_file.imageData)
            imgviz.io.imsave(out_img_file, img)
            data["images"].append(
                dict(
                    license=0,
                    url=None,
                    file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                    height=img.shape[0],
                    width=img.shape[1],
                    date_captured=None,
                    id=image_id,
                )
            )

            masks = {}  # for area
            segmentations = collections.defaultdict(list)  # for segmentation
            for shape in label_file.shapes:
                points = shape["points"]
                label = shape["label"]
                group_id = shape.get("group_id")
                shape_type = shape.get("shape_type", "polygon")
                mask = labelme.utils.shape_to_mask(
                    img.shape[:2], points, shape_type
                )

                if group_id is None:
                    group_id = uuid.uuid1()

                instance = (label, group_id)

                if instance in masks:
                    masks[instance] = masks[instance] | mask
                else:
                    masks[instance] = mask

                if shape_type == "rectangle":
                    (x1, y1), (x2, y2) = points
                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])
                    points = [x1, y1, x2, y1, x2, y2, x1, y2]
                if shape_type == "circle":
                    (x1, y1), (x2, y2) = points
                    r = np.linalg.norm([x2 - x1, y2 - y1])
                    # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                    # x: tolerance of the gap between the arc and the line segment
                    n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                    i = np.arange(n_points_circle)
                    x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                    y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                    points = np.stack((x, y), axis=1).flatten().tolist()
                else:
                    points = np.asarray(points).flatten().tolist()

                segmentations[instance].append(points)
            segmentations = dict(segmentations)

            for instance, mask in masks.items():
                cls_name, group_id = instance
                if cls_name not in class_name_to_id:
                    continue
                cls_id = class_name_to_id[cls_name]

                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                area = float(pycocotools.mask.area(mask))
                bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

                data["annotations"].append(
                    dict(
                        id=len(data["annotations"]),
                        image_id=image_id,
                        category_id=cls_id,
                        segmentation=segmentations[instance],
                        area=area,
                        bbox=bbox,
                        iscrowd=0,
                    )
                )

            if not self.noviz:
                viz = img
                if masks:
                    labels, captions, masks = zip(
                        *[
                            (class_name_to_id[cnm], cnm, msk)
                            for (cnm, gid), msk in masks.items()
                            if cnm in class_name_to_id
                        ]
                    )
                    viz = imgviz.instances2rgb(
                        image=img,
                        labels=labels,
                        masks=masks,
                        captions=captions,
                        font_size=15,
                        line_width=2,
                    )
                out_viz_file = osp.join(
                    self.output_dir, "Visualization", base + ".jpg"
                )
                imgviz.io.imsave(out_viz_file, viz)

        with open(osp.join(self.output_dir, "annotations.json"), "w") as f:
            json.dump(data, f)

# Revised and organized from https://doi.org/10.5281/zenodo.2738323 and https://github.com/ultralytics/JSON2YOLO .
class SegJSON:
    def __init__(self, json_dir='../coco/annotations/', save_dir='segdata/', use_segments=True, split_ratio=0.8):
        self.json_dir = json_dir
        self.use_segments = use_segments
        self.split_ratio = split_ratio
        self.save_dir = save_dir
        self.make_dirs()

    def convert_coco_json(self):
        save_dir = self.save_dir  # output directory

        # Import json
        for json_file in sorted(Path(self.json_dir).resolve().glob('*.json')):
            fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
            fn.mkdir()
            with open(json_file) as f:
                data = json.load(f)

            # Create image dict
            images = {'%g' % x['id']: x for x in data['images']}
            # Create image-annotations dict
            imgToAnns = defaultdict(list)
            for ann in data['annotations']:
                imgToAnns[ann['image_id']].append(ann)

            # Write labels file
            for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
                img = images['%g' % img_id]
                h, w, f = img['height'], img['width'], img['file_name']

                bboxes = []
                segments = []
                for ann in anns:
                    if ann['iscrowd']:
                        continue
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(ann['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue

                    cls = ann['category_id']
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if self.use_segments:
                        if len(ann['segmentation']) > 1:
                            s = self.merge_multi_segment(ann['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)

                # Write
                with open((fn / f[11:]).with_suffix('.txt'), 'a') as file:
                    for i in range(len(bboxes)):
                        line = *(segments[i] if self.use_segments else bboxes[i]),  # cls, box or segments
                        if line[0] is None:
                            continue
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')

        self.split_train_val()

    def make_dirs(self):
        # Create necessary directories
        save_dir = Path(self.save_dir)  # Change this to your desired output directory
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (save_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (save_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    def merge_multi_segment(segments):
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list = [[] for _ in range(len(segments))]

        def min_index(arr1, arr2):
            dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
            return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        for k in range(2):
            # forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s

    def delete_dsstore(path='../datasets'):
        # Delete apple .DS_store files
        from pathlib import Path
        files = list(Path(path).rglob('.DS_store'))
        print(files)
        for f in files:
            f.unlink()

    def split_train_val(self):
        # Function to split the dataset into train and val sets based on split ratio
        train_paths, val_paths = [], []  # Lists to store paths
        ann_path = os.path.join(self.save_dir, 'labels/annotations')
        anns = [os.path.splitext(i)[0] for i in os.listdir(ann_path)]
        random.shuffle(anns)
        split_idx = int(len(anns) * self.split_ratio)
        train_ids, val_ids = anns[:split_idx], anns[split_idx:]

        img_path = os.path.join(self.json_dir, 'JPEGImages')
        for img_id in os.listdir(img_path):
            img_name, _ = os.path.splitext(img_id)
            img_op = os.path.join(img_path, img_id)
            txt_op = os.path.join(ann_path, img_name + '.txt')
            if img_name in train_ids:
                fn = os.path.join(self.save_dir, 'labels/train/' + img_name + '.txt')
                img_n = os.path.join(self.save_dir, 'images/train', img_id)
                train_paths.append(fn)
            else:
                fn = os.path.join(self.save_dir, 'labels/val/' + img_name + '.txt')
                img_n = os.path.join(self.save_dir, 'images/val', img_id)
                val_paths.append(fn)

            shutil.copyfile(img_op, img_n)
            shutil.copyfile(txt_op, fn)

        with open(os.path.join(self.save_dir, 'train.txt'), 'w') as f:
            for path in train_paths:
                f.write(f"{path}\n")

        with open(os.path.join(self.save_dir, 'val.txt'), 'w') as f:
            for path in val_paths:
                f.write(f"{path}\n")









