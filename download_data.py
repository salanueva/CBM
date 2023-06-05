import argparse

import gdown
import os
import zipfile

from src.data.constants import DATA_ROOT

def update_data_root(args):

    with open("src/data/constants.py", "r") as f:
        constants = f.readlines()
    
    with open("src/data/constants.py", "w") as f:
        for line in constants:
            if line.startswith("DATA_ROOT"):
                f.write(f"DATA_ROOT = \"{args.data_dir}\"\n")
            else:
                f.write(line)


def downaload_zip_files(urls, data_path):

    for k, v in urls.items():
        output_file = os.path.join(data_path, k)
        gdown.download(v, output_file, quiet=False)
    

def uncompress_and_delete_zip_files(files, data_path):
    
    for f in files:
        with zipfile.ZipFile(os.path.join(data_path, f), 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.remove(os.path.join(data_path, f))
        


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path in which data will be stored."
    )

    args = parser.parse_args()
    return args



def main():

    args = parse_args()

    if args.data_dir is not None:
        data_path = args.data_dir
        update_data_root(args)
    else:
        data_path = DATA_ROOT
        
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    urls = {
        "annotations.zip": 'https://drive.google.com/uc?id=1TuTL5ijxAURDKt9Tw8bdlIMBPPY0DAgK',
        "captions.zip": 'https://drive.google.com/uc?id=1L4SKXhnmGUP3VDlCDWs30GtOlC_LXuGu',
        "splits.zip": 'https://drive.google.com/uc?id=13ONH63gM9jTNdvmHFJwSIXFKhfPdWE3C',
        "vocab.zip": 'https://drive.google.com/uc?id=1tku5duPS9u22QF2xwtQL8EDYlFk-laYb',
    }
    
    downaload_zip_files(urls, data_path)
    uncompress_and_delete_zip_files(urls.keys(), data_path)


if __name__ == "__main__":
    main()
