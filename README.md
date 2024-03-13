# crabcomm
This repository provides essential dataset preprocessing tools for YOLO object detection and other image classification models. Features include file conversion, sliding window cropping, image correction, and dataset balancing, designed to optimize image datasets for improved model training and accuracy.

## 'Crabcomm' package
Author:Xuan Gu, Date: 2024/2/26
### Main Functions Explained
First, all files should be stored in one folder, for example `crabcomm`. Next, you can use `from crabcomm.Preprocessing2 import *` to activate these functions.
#### 1. **crop_images**

**Function Description**: This function is designed to crop multiple smaller images from original images in a specified directory. It does so by applying a sliding window technique over the original images. The size of the cropped images, the step size of the sliding window, and whether to keep the original image names can be specified.

**Parameters**:
- `ori_path`: The directory path where the original images are stored.
- `crop_path`: The directory path where the cropped images will be saved.
- `crop_size`: The size of the cropped images in pixels.
- `step_size`: The step size of the sliding window in pixels.
- `keepname`: A boolean indicating whether to keep the original image names. If `True`, the names of the cropped images will be based on the original image names; otherwise, a uniform naming format will be used.
- `sample_rate`: The sampling rate for processing the original images, ranging from 0 to 1. For example, 0.5 means only 50% of the images in the original directory will be processed.

**Usage Example**:
```python
crop_images('path/to/original/images', 'path/to/cropped/images', crop_size=256, step_size=128, keepname=True, sample_rate=1.0)
```

#### 2. **class_count_xml**

**Function Description**: This function reads XML files to count and classify objects within images. It is typically used in datasets where annotations are stored in XML format, allowing for the analysis of object classes and their frequencies within a dataset.

**Parameters**:
- Specific parameters for this function are not detailed in the provided code snippet, but it would typically require paths to the XML files and potentially a list of class names to look for.

**Usage Example**:
```python
class_count_xml('path/to/xml/files')
```

#### 3. **class_count_txt**

**Function Description**: Similar to `class_count_xml`, this function reads from text files to count and classify objects within images. It's used when annotations are stored in a text-based format, facilitating the analysis of object classes and their distribution.

**Parameters**:
- Again, specific parameters are not provided, but it would generally need paths to the text files and possibly a list of class names.

**Usage Example**:
```python
class_count_txt('path/to/txt/files')
```

#### 4. **splitratio**

**Function Description**: This function is responsible for splitting a dataset into training and validation sets based on a specified ratio. It ensures that a dataset can be divided for the purpose of training and validating machine learning models, with separate subsets for each.

**Parameters**:
- The parameters for this function are not explicitly mentioned, but it would typically require the dataset path, the desired split ratio, and possibly parameters for random shuffling.

**Usage Example**:
```python
splitratio('path/to/dataset', train_ratio=0.8)
```

#### 5. **anchor_kmeans_plusplus**

**Function Description**: This function applies the K-means++ clustering algorithm to calculate optimal anchor box sizes for object detection models. By analyzing the widths and heights of bounding boxes in a dataset, it identifies the best sizes for anchor boxes that can improve model performance.

**Parameters**:
- `xml_dir`: The directory containing XML files with bounding box annotations.
- `output_dir`: The directory where outputs like the bounding box widths and heights, and the calculated anchor sizes will be saved.

**Usage Example**:
```python
anchor_kmeans_plusplus('path/to/xmls', 'path/to/output')
```

#### 6. **rotate_and_check_image_size**

**Function Description**:  
Rotates an image if necessary (when the width is less than the height) and checks if the image matches the desired dimensions (width and height). If the image matches the desired dimensions after any necessary rotation, it is saved back to the same path.

**Parameters**:  
- `image_path` (str): Path to the image file.
- `desired_width` (int, optional): Desired width of the image, default is 6000 pixels.
- `desired_height` (int, optional): Desired height of the image, default is 4000 pixels.

**Usage Example**:  
```python
image_path = "/path/to/image.jpg"
rotate_and_check_image_size(image_path, 6000, 4000)
```

---

#### 7. **flip_images**

**Function Description**:  
Iterates through all images in a specified folder, flipping each image to match the desired dimensions. This function utilizes `rotate_and_check_image_size` to ensure each image meets the dimension criteria.

**Parameters**:  
- `folder_path` (str): Path to the folder containing images to be flipped.
- `desired_width` (int, optional): Desired width of the images after flipping, default is 6000 pixels.
- `desired_height` (int, optional): Desired height of the images after flipping, default is 4000 pixels.

**Usage Example**:  
```python
folder_path = "/path/to/images"
flip_images(folder_path, 6000, 4000)
```

---

#### 8. **balance_dataset**

**Function Description**:  
Balances a dataset by copying images and their annotation files to new directories according to specified class occurrence conditions. This function helps ensure that each class is represented within certain limits, preventing class imbalance.

**Parameters**:  
- `img_dir` (str): Source directory containing images.
- `txt_dir` (str): Source directory containing corresponding annotation text files.
- `out_img_dir` (str): Target directory for copied images.
- `out_txt_dir` (str): Target directory for copied annotation text files.
- `target_classes` (list of int): List of class IDs to be considered for balancing.
- `class_counts_path` (str): Path to the text file containing initial class counts.
- `classes` (list of str): List of class names corresponding to the class IDs.

**Usage Example**:  
```python
img_dir = "/source/images"
txt_dir = "/source/annotations"
out_img_dir = "/target/images"
out_txt_dir = "/target/annotations"
target_classes = [1, 2, 3]
class_counts_path = "/path/to/class_counts.txt"
classes = ["Class1", "Class2", "Class3"]

balance_dataset(img_dir, txt_dir, out_img_dir, out_txt_dir, target_classes, class_counts_path, classes)
```

---

#### 9. **read_classes_from_file**

**Function Description**:  
Reads class names from a given text file where each line contains a single class name. This function returns a list of class names.

**Parameters**:  
- `filepath` (str): Path to the text file containing class names.

**Returns**:  
- A list of class names.

**Usage Example**:  
```python
filepath = "/path/to/classes.txt"
classes = read_classes_from_file(filepath)
```

---

#### 10. **crop_objects_from_images**

**Function Description**:  
Crops objects from images based on annotations in corresponding XML files and saves the cropped images into a specified directory. This function is useful for creating datasets for object detection tasks.

**Parameters**:  
- `img_path` (str): Source directory containing images.
- `xml_path` (str): Source directory containing XML annotation files.
- `obj_img_path` (str): Target directory for saving cropped images.
- `classes_file` (str): Path to the text file containing class names.

**Usage Example**:  
```python
img_path = "/path/to/images"
xml_path = "/path/to/xmls"
obj_img_path = "/path/to/cropped_images"
classes_file = "/path/to/classes.txt"

crop_objects_from_images(img_path, xml_path, obj_img_path, classes_file)
```

---

#### 11. **create_map**

**Function Description**:  
Creates a mapping file for class names to class IDs, saving it as a JSON string. This function facilitates the association of class names with their respective IDs in a structured format.

**Parameters**:  
- `cate` (list of str): A list of class names.
- `map_txt_path` (str): The path where the class mapping file will be saved.

**Usage Example**:  
```python
cate = ["Class1", "Class2", "Class3"]
map_txt_path = "/path/to/map.txt"

create_map(cate, map_txt_path)
```

---

#### 12. **crab_map_make**

**Function Description**:  
Generates training file mappings and creates a class mapping file. This function is designed to prepare data and mappings for training machine learning models.

**Parameters**:  
- `ori_folder` (str): The path to the original folder containing class folders.
- `train_file_save_path` (str): The path where the training files should be saved.
- `classes` (list of str): A list of class names.
- `map_txt_path` (str): The path where the class mapping file will be saved.

**Usage Example**:  
```python
ori_folder = "/path/to/original_folder"
train_file_save_path = "/path/to/save_training_files"
classes = ["Class1", "Class2", "Class3"]
map_txt_path = "/path/to/class_map.txt"

crab_map_make(ori_folder, train_file_save_path, classes, map_txt_path)
```

---

#### 13. **ImageProcessor**

**Function Description:**

This class processes images based on the annotations provided in TXT files. It performs operations such as undistorting images, reading rectangles from TXT files, applying perspective transforms, and saving the transformed images.

**Parameters:**
- `input_txt_folder`: Path to the folder containing TXT files with annotations.
- `input_image_folder`: Path to the folder containing the original images.
- `output_image_folder`: Path to the folder where processed images will be saved.

**Usage Example:**
```python
input_txt_folder = 'path/to/txt/folder'
input_image_folder = 'path/to/image/folder'
output_image_folder = 'path/to/output/folder'
processor = ImageProcessor(input_txt_folder, input_image_folder, output_image_folder)
processor.process_images()
```

---

#### 14. **TxtToXmlConverter**

**Function Description:**

This class converts annotations from TXT format to XML format, following the PASCAL VOC structure. It's useful for adapting datasets to different object detection frameworks.

**Parameters:**
- `input_dir`: Directory containing the TXT files to be converted.
- `output_dir`: Directory where the XML files will be saved.
- `img_width`: Width of the images referenced in the TXT files.
- `img_height`: Height of the images referenced in the TXT files.
- `categ_path`: Path to the file listing the categories/classes.

**Usage Example:**
```python
input_dir = 'path/to/txt/files'
output_dir = 'path/to/xml/files'
img_width = 640
img_height = 640
categ_path = 'path/to/categories.txt'
converter = TxtToXmlConverter(input_dir, output_dir, img_width, img_height, categ_path)
converter.convert_all_txt_to_xml()
```

---

#### 15. **XMLToTxtConverter**

**Function Description:**

This class converts annotations from XML format to TXT format, suitable for YOLO and other object detection models. It supports filtering objects by class and difficulty level.

**Parameters:**
- `categ_path`: Path to the file containing the category names.
- `txt_path`: Path to the directory where the TXT files will be saved.
- `xml_path`: Path to the directory containing the XML files to convert.
- `img_path`: Path to the directory containing the images referenced in the XML files.

**Usage Example:**
```python
categ_path = 'path/to/categories.txt'
xml_path = 'path/to/xml/files'
txt_path = 'path/to/output/txt/files'
img_path = 'path/to/images'
converter = XMLToTxtConverter(categ_path, txt_path, xml_path, img_path)
converter.convert_all()
```

---

#### 16. **LabelmeToJSON**

#### Revised and organized from https://github.com/labelmeai/labelme/blob/12d425b956878132566d243b4d9f6f3af33ec810/examples/bbox_detection/labelme2voc.py#L70.
#### Wada, K. Labelme: Image Polygonal Annotation with Python [Computer software]. https://doi.org/10.5281/zenodo.5711226

**Function Description:**

The `LabelmeToJSON` class provides functionality to convert Labelme annotations to a COCO JSON format. It reads Labelme annotations from a specified input directory, processes the data, and generates a COCO JSON file in the output directory. The class also handles the conversion of image data, annotations, and categories.

**Parameters:**
- `input_dir` (str): Path to the input directory containing Labelme annotations.
- `output_dir` (str): Path to the output directory where the COCO JSON file will be saved.
- `labels_file` (str): Path to the file containing class labels.
- `noviz` (bool): Flag indicating whether visualization images should be generated. Default is False.

**Usage Example:**
```python
labelme_converter = LabelmeToJSON(input_dir='labelme_annotations/', output_dir='coco_json/', labels_file='class_labels.txt', noviz=False)
labelme_converter.convert()
```

---

#### 17. **SegJSON**

#### Revised and organized from https://doi.org/10.5281/zenodo.2738323 and https://github.com/ultralytics/JSON2YOLO .

**Function Description:**

The `SegJSON` class is designed to process annotation data in COCO format and convert it for segmentation tasks. This class supports splitting the dataset into training and validation sets. It creates corresponding directories for images and labels within a specified output directory. Moreover, it generates `train.txt` and `val.txt` files listing the relative paths to the images for training and validation, respectively.

**Parameters:**
- `json_dir` (str): Directory path where the JSON files are located. Default is '../coco/annotations/'.
- `save_dir` (str): Directory path where the converted data will be saved. Default is 'segdata/'.
- `use_segments` (bool): Flag to indicate whether to use segments during conversion. Default is True.
- `split_ratio` (float): Ratio to split the dataset into train and validation sets. Default is 0.8.

**Usage Example:**
```python
seg_json = SegJSON(json_dir='../custom_annotations/', save_dir='output_data/', use_segments=True, split_ratio=0.8)
seg_json.convert_coco_json()
```

These classes provide a comprehensive set of tools for preparing and transforming image annotations for various purposes in computer vision projects, including object detection and image processing tasks.