""" Helper Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
from __future__ import annotations

import os
from typing import Any, Callable

from torchvision import datasets

from lightly.data._image import DatasetFolder
from lightly.data._video import VideoDataset

try:
    from lightly.data._video import VideoDataset

    """ Helper Functions """

    VIDEO_DATASET_AVAILABLE = False
except Exception as e:
    VIDEO_DATASET_AVAILABLE = False
    VIDEO_DATASET_ERRORMSG = e


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mpg", ".hevc", ".m4v", ".webm", ".mpeg")


def _dir_contains_videos(root: str, extensions: tuple[str, ...]) -> bool:
    """Checks whether the directory contains video files.

    Args:
        root: Root directory path.
        extensions: Tuple of valid video file extensions.

    Returns:
        True if the root directory contains video files, False otherwise.
    """
    with os.scandir(root) as scan_dir:
        return any(f.name.lower().endswith(extensions) for f in scan_dir)


def _contains_videos(root: str, extensions: tuple[str, ...]) -> bool:
    """Checks whether the directory or any subdirectory contains video files.

    Args:
        root: Root directory path.
        extensions: Tuple of valid video file extensions.

    Returns:
        True if the root directory or any subdirectory contains video files, False otherwise.
    """
    for subdir, _, _ in os.walk(root):
        if _dir_contains_videos(subdir, extensions):
            return True
    return False


def _is_lightly_output_dir(dirname: str) -> bool:
    """Checks whether the directory is a lightly_output directory.

    Args:
        dirname: Directory name to check.

    Returns:
        True if the directory name is "lightly_outputs", False otherwise.
    """
    return "lightly_outputs" in dirname


def _contains_subdirs(root: str) -> bool:
    """Checks whether the directory contains subdirectories.

    Args:
        root: Root directory path.

    Returns:
        True if the root directory contains subdirectories (excluding "lightly_outputs"), False otherwise.
    """
    with os.scandir(root) as scan_dir:
        return any(not _is_lightly_output_dir(f.name) for f in scan_dir if f.is_dir())


def _load_dataset_from_folder(
    root: str,
    transform: Callable[[Any], Any],
    is_valid_file: Callable[[str], bool] | None,
    tqdm_args: dict[str, Any] | None,
    num_workers_video_frame_counting: int = 0,
) -> datasets.VisionDataset:
    """Initializes a dataset from a folder.

    This function determines the appropriate dataset type based on the contents of the root directory
    and returns the corresponding dataset object.

    Args:
        root: Root directory path.
        transform: Composed image transformations to be applied to the dataset.
        is_valid_file: Optional function to determine valid files.
        tqdm_args: Optional dictionary of arguments for tqdm progress bar.
        num_workers_video_frame_counting: Number of workers for video frame counting.

    Returns:
        A dataset object (VideoDataset, ImageFolder, or DatasetFolder) based on the directory contents.

    Raises:
        ValueError: If the specified dataset directory doesn't exist or if videos are present
                but VideoDataset is not available.
    """
    if not os.path.exists(root):
        raise ValueError(f"The input directory {root} does not exist!")

    # Perform the required directory and extension scan only once
    contains_videos, contains_subdirs = _fast_contains_videos_and_subdirs(
        root, VIDEO_EXTENSIONS
    )

    if contains_videos and not VIDEO_DATASET_AVAILABLE:
        raise ValueError(
            f"The input directory {root} contains videos "
            "but the VideoDataset is not available. "
            "Make sure you have installed the right "
            "dependencies. The error from the imported "
            f"module was: {VIDEO_DATASET_ERRORMSG}"
        )

    if contains_videos:
        return VideoDataset(
            root,
            extensions=VIDEO_EXTENSIONS,
            transform=transform,
            is_valid_file=is_valid_file,
            tqdm_args=tqdm_args,
            num_workers=num_workers_video_frame_counting,
        )
    elif contains_subdirs:
        return datasets.ImageFolder(
            root, transform=transform, is_valid_file=is_valid_file
        )
    else:
        return DatasetFolder(
            root,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            is_valid_file=is_valid_file,
        )


def _fast_contains_videos_and_subdirs(root: str, extensions: tuple[str, ...]):
    """Checks whether directory tree contains videos and/or subdirs (non-lightly_outputs)."""
    contains_videos = False
    contains_subdirs = False
    stack = [root]
    while stack and (not contains_videos or not contains_subdirs):
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    if entry.is_file():
                        if entry.name.lower().endswith(extensions):
                            contains_videos = True
                            if contains_subdirs or current == root:
                                # Early out if both true or at top level and found videos
                                return contains_videos, contains_subdirs
                    elif entry.is_dir():
                        if not _is_lightly_output_dir(entry.name):
                            # Only check for subdirs at the top-level
                            if current == root:
                                contains_subdirs = True
                                if contains_videos:
                                    return contains_videos, contains_subdirs
                            # Always descend for videos
                            stack.append(entry.path)
        except PermissionError:
            continue
    return contains_videos, contains_subdirs
