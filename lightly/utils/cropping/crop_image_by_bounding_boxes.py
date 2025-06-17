import os
import os.path
import warnings
from typing import List

from PIL import Image
from tqdm import tqdm

from lightly.data import LightlyDataset
from lightly.utils.bounding_box import BoundingBox


def crop_dataset_by_bounding_boxes_and_save(
    dataset: LightlyDataset,
    output_dir: str,
    bounding_boxes_list_list: List[List[BoundingBox]],
    class_indices_list_list: List[List[int]],
    class_names: List[str] = None,
) -> List[List[str]]:
    """Crops all images in a dataset by the bounding boxes and saves them in the output dir

    Args:
        dataset:
            The dataset with the images to be cropped. Must contain M images.
        output_dir:
            The output directory to saved the cropped images to.
        bounding_boxes_list_list:
            The bounding boxes of the detections for each image. Must have M sublists, one for each image.
            Each sublist contains the bounding boxes for each detection, thus N_m elements.
        class_indices_list_list:
            The object class ids of the detections for each image. Must have M sublists, one for each image.
            Each sublist contains the bounding boxes for each detection, thus N_m elements.
        class_names:
            The names of the classes, used to map the class id to the class name.


    Returns:
        The filepaths to all saved cropped images. Has M sublists, one for each image.
        Each sublist contains the filepath of the crop each detection, thus N_m elements.

    """
    filenames_images = dataset.get_filenames()
    n_imgs = len(filenames_images)
    if n_imgs != len(bounding_boxes_list_list) or n_imgs != len(
        class_indices_list_list
    ):
        raise ValueError(
            "There must be one bounding box and class index list for each image in the datasets,"
            "but the lengths dont align."
        )

    cropped_image_filepath_list_list: List[List[str]] = []
    print(f"Cropping objects out of {n_imgs} images...")

    dir_cache = set()  # Avoid multiple mkdirdirs for the same output

    # Use enumerate and zip to avoid tuple unpacking in for loop
    for img_idx, (filename_image, class_indices, bounding_boxes) in enumerate(
        tqdm(
            zip(filenames_images, class_indices_list_list, bounding_boxes_list_list),
            total=n_imgs,
        )
    ):
        if len(class_indices) != len(bounding_boxes):
            warnings.warn(
                UserWarning(
                    f"Length of class indices ({len(class_indices)}) does not equal length of bounding boxes"
                    f"({len(bounding_boxes)}). This is an error in the input arguments. "
                    f"Skipping this image {filename_image}."
                )
            )
            continue

        # Get image full path (Disk I/O is unavoidable)
        filepath_image = dataset.get_filepath_from_filename(filename_image)
        filepath_image_base, image_extension = os.path.splitext(filepath_image)

        # Use string ops, avoid repeated replace if not needed
        filepath_out_dir = os.path.join(output_dir, filename_image)
        if filepath_out_dir.endswith(image_extension):
            filepath_out_dir = filepath_out_dir[: -len(image_extension)]

        if filepath_out_dir not in dir_cache:
            os.makedirs(filepath_out_dir, exist_ok=True)
            dir_cache.add(filepath_out_dir)

        # Open image (disk I/O, unavoidable)
        image = Image.open(filepath_image)
        w, h = image.size  # Only compute once per image

        cropped_images_filepaths = []
        for index, (class_index, bbox) in enumerate(zip(class_indices, bounding_boxes)):
            if class_names:
                class_name = class_names[class_index]
            else:
                class_name = f"class{class_index}"
            cropped_image_last_filename = f"{index}_{class_name}{image_extension}"
            cropped_image_filepath = os.path.join(
                filepath_out_dir, cropped_image_last_filename
            )

            # Integer cropping, avoid per-element generator
            crop_box = (
                int(w * bbox.x0),
                int(h * bbox.y0),
                int(w * bbox.x1),
                int(h * bbox.y1),
            )
            cropped_image = image.crop(crop_box)
            cropped_image.save(cropped_image_filepath)

            cropped_image_filename = os.path.join(
                filename_image[: -len(image_extension)]
                if filename_image.endswith(image_extension)
                else filename_image,
                cropped_image_last_filename,
            )
            cropped_images_filepaths.append(cropped_image_filename)

        cropped_image_filepath_list_list.append(cropped_images_filepaths)

    return cropped_image_filepath_list_list
