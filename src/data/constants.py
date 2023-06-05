# Paths for data....

DATA_ROOT = "./data"

FEAT_FOLDER = "features"
CAP_FOLDER = {
    "human": {
        "train": "captions/train/formatted_captions_train2014.json",
        "val": "captions/val/formatted_captions_val2014.json",
    },
    "auto_contaminated": {
        "train": "captions/train/pred.contaminated_coco_caption.train.beam5.max20.odlabels_coco_format.json",
        "val": "captions/val/pred.contaminated_coco_caption.val.beam5.max20.odlabels_coco_format.json",
        "test": "captions/test/pred.contaminated_coco_caption.test.beam5.max20.odlabels_coco_format.json",
    },
    "auto": {
        "train": "captions/train/pred.coco_caption.train.beam5.max20.odlabels_coco_format.json",
        "val": "captions/val/pred.coco_caption.val.beam5.max20.odlabels_coco_format.json",
        "test": "captions/test/pred.coco_caption.test.beam5.max20.odlabels_coco_format.json",
    }
}

#################
# OKVQA DATASET #
#################

OKVQA_TRAIN_QUESTION_FILE = "annotations/okvqa/OpenEnded_mscoco_train2014_questions.json"
OKVQA_TEST_QUESTION_FILE = "annotations/okvqa/OpenEnded_mscoco_val2014_questions.json"
OKVQA_TRAIN_ANN_FILE_V1_0 = "annotations/okvqa/mscoco_train2014_annotations_v1.0.json"
OKVQA_TEST_ANN_FILE_V1_0 = "annotations/okvqa/mscoco_val2014_annotations_v1.0.json"
OKVQA_TRAIN_ANN_FILE_V1_1 = "annotations/okvqa/mscoco_train2014_annotations_v1.1.json"
OKVQA_TEST_ANN_FILE_V1_1 = "annotations/okvqa/mscoco_val2014_annotations_v1.1.json"

OKVQA_SPLIT_FILE = "splits/okvqa_train_split.npz"
OKVQA_SPLIT_T5 = {
    "train": "splits/splitted_train_ids.json",
    "val": "splits/splitted_val_ids.json"
}

OKVQA_ANS_IDX2WORD_V1_0 = f"vocab/okvqa_answer_vocab_idx2word_v1.0.json"
OKVQA_ANS_WORD2IDX_V1_0 = f"vocab/okvqa_answer_vocab_word2idx_v1.0.json"

OKVQA_ANS_IDX2WORD_V1_1 = f"vocab/okvqa_answer_vocab_idx2word_v1.1.json"
OKVQA_ANS_WORD2IDX_V1_1 = f"vocab/okvqa_answer_vocab_word2idx_v1.1.json"


###############
# VQA DATASET #
###############

VQA_TRAIN_QUESTION_FILE = "annotations/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
VQA_VAL_QUESTION_FILE = "annotations/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_TRAIN_ANN_FILE = "annotations/vqa/v2_mscoco_train2014_annotations.json"
VQA_VAL_ANN_FILE = "annotations/vqa/v2_mscoco_val2014_annotations.json"

VQA_ANS_IDX2WORD = f"vocab/vqa_answer_vocab_idx2word_v2.json"
VQA_ANS_WORD2IDX = f"vocab/vqa_answer_vocab_word2idx_v2.json"

VQA_TRAIN_SPLIT_FILE = "splits/vqa_train_split.npz"
VQA_MINIVAL_SPLIT_FILE = "splits/vqa_minival_split.json"
VQA_NOMINIVAL_SPLIT_FILE = "splits/vqa_no_minival_split.json"
