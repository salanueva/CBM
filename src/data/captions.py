import json
import os

from data.constants import DATA_ROOT, CAP_FOLDER


def load_human(split):
    with open(os.path.join(DATA_ROOT, CAP_FOLDER["human"][split]), "r") as f:
        gt_data = json.load(f)
    return gt_data
    

def load_oscar(split, contaminated=False):
    if contaminated:
        with open(os.path.join(DATA_ROOT, CAP_FOLDER["auto_contaminated"][split]), "r") as f:
            oscar_data = json.load(f)
    else:
        with open(os.path.join(DATA_ROOT, CAP_FOLDER["auto"][split]), "r") as f:
            oscar_data = json.load(f)
    return {elem["image_id"]: elem["caption"] for elem in oscar_data}


def load_all_captions(cap_type):
    if cap_type == "oscar":
        train_data = load_oscar("train")
        dev_data = load_oscar("val")
        test_data = load_oscar("test")
        return {**train_data, **dev_data, **test_data}
    elif cap_type == "human":
        train_data = load_human("train")
        dev_data = load_human("val")
        return {**train_data, **dev_data}
    else:
        raise NotImplementedError