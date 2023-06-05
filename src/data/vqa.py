import numpy as np
import os
from pathlib import Path
import pickle
import torch
import torchvision

from data.constants import *
from data.captions import *


class Vqa(torchvision.datasets.vision.VisionDataset):

    def __init__(self, root, q_file, ann_file, feat_path=None, cap_type=None):
        """
        :param root: Path where all files can be found.
        :param q_file: Path to json question file.
        :param ann_file: Path to json annotation file.
        :param feat_path: Path to precomputed features of image regions.
        :param cap_type: Which captions are we going to use? Two options: "oscar" or dense".
        """
        super(Vqa, self).__init__(root)
        
        with open(os.path.join(self.root, q_file), "r") as f:
            questions = json.load(f)

        with open(os.path.join(self.root, ann_file), "r") as f:
            annotations = json.load(f)

        captions = None
        if cap_type is not None:
            captions = load_all_captions(cap_type)
        
        features = None
        if feat_path is not None and os.path.isdir(Path(os.path.join(self.root, feat_path))):
            features = feat_path
        elif feat_path is not None:
            with open(os.path.join(self.root, feat_path), "rb") as f:
                features = pickle.load(f)
        
        img_ids = {str(q["image_id"]): 1 for q in questions["questions"]}
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
        :return: tuple (image, caption, question, answers). answers are 10 possible answers
        """

        question = self.questions["questions"][index]
        annotation = self.annotations["annotations"][index]

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

        # TODO: Change...
        question_text = "question: " + question["question"]
        caption_text = "caption: " + caption

        return features, caption_text, question_text, list(answers)

    def __len__(self):
        return self.length
