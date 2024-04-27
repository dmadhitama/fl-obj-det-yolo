import os
import random
from typing import List, Tuple, Dict, Any
import augment
import cv2
from tqdm import tqdm
import argparse
import shutil
import re

import sys
sys.path.append(".")
from datasets import augment 

class Dataset:
    def __init__(self, directory: str):
        self.directory: str = directory
        self.file_pairs: List[Tuple[str, str]] = self.list_file_pairs()
        self.category_id2name: Dict[int, str] = {0: 'table', 1: 'ref', 2: 'object'}
        self.category_name2id: Dict[int, str] = {'table': 0, 'ref': 1, 'object': 2}

    def list_file_pairs(self) -> List[Tuple[str, str]]:
        """Returns a list of file pairs.

        Returns:
            A list of file pairs. Each pair is a tuple of two strings.
        """
        file_dict: Dict[str, List[str]] = {}
        for root, _, files in os.walk(self.directory):
            for file in files:
                base, ext = os.path.splitext(file)
                if ext in (".txt", ".jpg"):
                    file_dict.setdefault(base, []).append(os.path.join(root, file))

        file_pairs: List[Tuple[str, str]] = []
        for base, file_list in file_dict.items():
            if len(file_list) == 2:
                file_pairs.append((file_list[0], file_list[1]))
        return file_pairs

    def split_data(
            self, 
            test_size: float = 0.2
        ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Splits the data into training and test sets.

        Args:
            test_size: The fraction of the data to be used for testing.

        Returns:
        A tuple containing the training and test sets. Each set is a list of tuples, where each tuple is a file pair.
        
        """
        random.shuffle(self.file_pairs)
        split_index = int(len(self.file_pairs) * test_size)
        return self.file_pairs[split_index:], self.file_pairs[:split_index]
    
    def augment_file_pairs(
            self, 
            filename_pairs: Tuple[List[Tuple[str, str]], List[Tuple[str, str]]], 
            n_augments: int, 
            augmented_dir: str = "augments",
            annotation_format: str = "yolo",
        ) -> None:
        """Augments the file pairs.

        Args:
            filename_pairs: Each set is a list of tuples, where each tuple is a file pair.
            n_augments (int): The number of augmentations to perform.
            augmented_dir (str, optional): The directory to save the augmented files. Defaults to "augments".
            annotation_format (str, optional): The format of the annotation file. Defaults to "coco".
            
        """
        # Create the augmented directory
        if not os.path.exists(augmented_dir):
            os.makedirs(augmented_dir)
        if annotation_format == "yolo":
            convert_to_yolo = True
        else:
            convert_to_yolo = False

        # Augment the file pairs
        print(f"Augmenting to {augmented_dir}...")
        for pair in tqdm(filename_pairs):
            # image_filename = pair[0].split("/")[-1]
            # annot_filename = pair[1].split("/")[-1]
            image_filename = re.split("/|\\\\", pair[0])[-1] # handle OS Windows paths
            annot_filename = re.split("/|\\\\", pair[1])[-1]
            transformed = augment.augment_a_file(
                pair,
                self.category_id2name,
                n_augments=n_augments,
                convert_to_yolo=convert_to_yolo,
            )
            
            for idx, t in enumerate(transformed):
                # Define augmented filename
                suffix = str(idx).zfill(4)
                aug_image_filename = image_filename.replace(".jpg", suffix+".jpg")
                aug_annot_filename = annot_filename.replace(".txt", suffix+".txt")
                # Write the augmented image to the directory
                cv2.imwrite(
                    os.path.join(augmented_dir, aug_image_filename), 
                    t["image"]
                )
                # Write the augmented annotation to the directory
                categories = [self.category_id2name[x] for x in t["category_ids"]]
                assert len(categories) == len(t["bboxes"])
                with open(os.path.join(augmented_dir, aug_annot_filename), "w") as f:
                    for i in range(len(categories)):
                        if convert_to_yolo:
                            f.write(
                                f"{t['category_ids'][i]}\t{t['bboxes'][i][0]}\t{t['bboxes'][i][1]}\t{t['bboxes'][i][2]}\t{t['bboxes'][i][3]}\t\n"
                            )
                        else:
                            f.write(
                                f"{t['bboxes'][i][0]},{t['bboxes'][i][1]},{t['bboxes'][i][2]},{t['bboxes'][i][3]},{categories[i]}\n"
                            )
            # Copy the original image and annotation to the new directory
            shutil.copy(pair[0], os.path.join(augmented_dir, image_filename))
            if convert_to_yolo:
                # Modify the original annotation to yolo format and write it to the new directory
                with open(pair[1], "r") as f:
                    lines = f.readlines()
                img = cv2.imread(pair[0])
                img_h, img_w, _ = img.shape
                with open(os.path.join(augmented_dir, annot_filename), "w") as f:
                    for line in lines:
                        bbox = line.strip().split(",")[:4]
                        bbox = [float(bb) for bb in bbox]
                        norm_bbox = augment.convert_and_normalize_bbox(
                            bbox,
                            (img_h, img_w),
                        )
                        label = line.strip().split(",")[-1]
                        label = self.category_name2id[label]
                        f.write(f"{label}\t{norm_bbox[0]}\t{norm_bbox[1]}\t{norm_bbox[2]}\t{norm_bbox[3]}\n")
            else:
                shutil.copy(pair[1], os.path.join(augmented_dir, annot_filename))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-directory", help="The directory to search for files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="The fraction of the data to be used for testing.")
    args = parser.parse_args()
    # Call the function to list the file pairs
    data = Dataset(args.target_directory)
    # pairs = data.file_pairs
    pairs_tr, pairs_tt = data.split_data(args.test_size)
    
    # Augment each pair of training and test data
    data.augment_file_pairs(pairs_tr, n_augments=3, augmented_dir="augments/train")
    data.augment_file_pairs(pairs_tt, n_augments=3, augmented_dir="augments/test")
    
    print("Done.")