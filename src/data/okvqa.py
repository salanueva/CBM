import numpy as np
import os
from pathlib import Path
import pickle
import random
import torch
import torchvision

from data.constants import *
from data.captions import *


class OkVqa(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, q_file, ann_file, feat_path=None, cap_type=None, ans_type="stemmed", is_tiny=False):
        """
        :param root: Path where all files can be found.
        :param q_file: Path to json quesiton file.
        :param ann_file: Path to json annotation file.
        :param feat_path: Path to precomputed features of image regions.
        :param cap_type: Which captions are we going to use? Two options: "oscar" or dense".
        :param ans_type: returned answer type. Two options: "raw" or "stemmed".
        :param is_tiny: if true only the first 10 instances are going to be loaded (for debugging purposes).
        """
        super(OkVqa, self).__init__(root)

        if ans_type == "raw":
            self.ans_type = "raw"
        else:
            self.ans_type = "stemmed"

        with open(os.path.join(self.root, q_file), "r") as f:
            questions = json.load(f)

        with open(os.path.join(self.root, ann_file), "r") as f:
            annotations = json.load(f)

        captions = None
        if cap_type is not None:
            captions = load_all_captions(cap_type)
            # Remove this line to apply data augmentation if multiple captions are given per instance (list of strings instead of a string)
            captions = {k: (random.choice(v) if isinstance(v, list) else v) for k, v in captions.items()}
        
        features = None
        if feat_path is not None and os.path.isdir(Path(feat_path)):
            features = feat_path
        elif feat_path is not None:
            with open(feat_path, "rb") as f:
                features = pickle.load(f)

        if is_tiny:
            questions["questions"] = questions["questions"][:10]
            annotations["annotations"] = annotations["annotations"][:10]
        
        # Remove data that won't be used
        img_ids = {str(q["image_id"]): 0 for q in questions["questions"]}
        
        if captions is not None:
            captions = {k: v for k, v in captions.items() if k in img_ids}

        self.img_ids = img_ids
        self.features = features
        self.questions = questions
        self.annotations = annotations
        self.captions = captions

        self.length = len(self.questions["questions"])

    def __getitem__(self, index):
        """
        :param index: index
        :return: tuple (image_features, caption, question, answers). answers are 10 possible answers
        """
        
        question = self.questions["questions"][index]
        annotation = self.annotations["annotations"][index]
  
        if self.ans_type == "raw":
            answers = [ans["raw_answer"] for ans in annotation["answers"]]
        else:
            answers = [ans["answer"] for ans in annotation["answers"]]

        img_id = str(question["image_id"])

        if self.features is not None and os.path.isdir(self.features):
            fpath = os.path.join(
                self.features, "train2014", f"COCO_train2014_{'0' * (12 - len(img_id)) + img_id}.npy"
            )
            if not os.path.isfile(fpath):
                fpath = os.path.join(
                    self.features, "val2014", f"COCO_val2014_{'0' * (12 - len(img_id)) + img_id}.npy"
                )
            fdata = np.load(fpath)
            features = torch.tensor(fdata[:36, :])
        else:
            features = torch.tensor(self.features[img_id]) if self.features is not None else 0

        caption = self.captions[img_id] if self.captions is not None else 0

        if isinstance(caption, list):
            caption = random.choice(caption)
        
        # TODO: Change
        question_text = question["question"]
        caption_text = caption
    
        return features, caption_text, question_text, list(answers)

    def __len__(self):
        return self.length
