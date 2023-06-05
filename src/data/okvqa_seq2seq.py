from pathlib import Path
from typing import Dict
import json
import os
import random

import torch
from torch.utils.data import Dataset

from data.captions import load_all_captions
from data.constants import OKVQA_SPLIT_T5


def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    tokenizer.padding_side = padding_side
    
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)

    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqOkVqaDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        root,
        q_file,
        ann_file,
        cap_type,
        max_source_length,
        max_target_length,
        ans_type="stemmed",
        split="train",
        n_obs=None,
        train_val_splitted=False
    ):
        super().__init__()

        self.root = root
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.tokenizer = tokenizer
        self.split = split

        if ans_type == "raw":
            self.ans_type = "raw"
        else:
            self.ans_type = "stemmed"

        with open(os.path.join(self.root, q_file), "r") as f:
            questions = json.load(f)

        with open(os.path.join(self.root, ann_file), "r") as f:
            annotations = json.load(f)
        
        if self.ans_type == "raw":
            answers = [[ans["raw_answer"] for ans in annotation["answers"]] for annotation in annotations["annotations"]]
        else:
            answers = [[ans["answer"] for ans in annotation["answers"]] for annotation in annotations["annotations"]]
        
        if train_val_splitted and self.split in ["train", "val"]:
            
            with open(OKVQA_SPLIT_T5[self.split], "r") as f:
                split_ids = json.load(f)
            
            questions = [questions[idx] for idx in split_ids]
            answers = [questions[idx] for idx in split_ids]
        
        correct_answers = []
        partial_answers = []
        for ans_set in answers:
            unique_answers = list(set(ans_set))
            c_ans, p_ans = [], []
            for u_ans in unique_answers:
                n = ans_set.count(u_ans)
                if n > 2:
                    c_ans.append(u_ans)
                else:
                    p_ans.append(u_ans)
            correct_answers.append("\t".join(c_ans))
            partial_answers.append("\t".join(p_ans))

        captions = load_all_captions(cap_type)
        # Remove this line to apply data augmentation if multiple captions are given per instance (list of strings instead of a string)
        captions = {k: (random.choice(v) if isinstance(v, list) else v) for k, v in captions.items()}
        
        self.src_lines = [f"caption: {captions[str(q['image_id'])]}; question: {q}" for q in questions["questions"]]
        self.tgt_lines = correct_answers
        self.tgt_partial_lines = partial_answers
        
        if not train_val_splitted:
            if self.split == "train":
                self.valid_idx = self.instance_ids_with_correct_answers()
            else:
                self.valid_idx = list(range(len(self.src_lines)))
            
        if n_obs is not None:
            self.valid_idx = self.valid_idx[:n_obs]
    

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]
    
    
    def instance_ids_with_correct_answers(self):
        valid_idx = [i for i, tgt in enumerate(self.tgt_lines) if tgt != ""]
        return valid_idx

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        index = self.valid_idx[idx]
        
        source_line = self.src_lines[index]
        tgt_line = self.tgt_lines[index]
        tgt_partial_line = self.tgt_partial_lines[index]
        
        tgt_answers = tgt_line.split("\t")
        if len(tgt_answers) > 1:
            tgt_answer = random.choice(tgt_answers)
        else:
            tgt_answer = tgt_answers[0]
        
        assert len(tgt_answer.split("\t")) == 1, f"more than one answer found for index {index}"
        assert source_line, f"empty source line for index {index}"
        assert tgt_line or tgt_partial_line, f"no targets for index {index}"

        # TODO: CHANGE THIS
        # if self.split == "train":
        #    assert tgt_line, f"empty target line for index {index}"

        # Pad source and target to the right
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length, "right")
        target_inputs = encode_line(self.tokenizer, tgt_answer, self.max_target_length, "right")

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
            "correct_answers_str": tgt_line,
            "partially_correct_answers_str": tgt_partial_line
        }
    
    def __len__(self):
        return len(self.valid_idx)

    def collate_fn(self, batch):
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        
        tgt_pad_token_id = self.tokenizer.pad_token_id
        src_pad_token_id = self.tokenizer.pad_token_id
        
        y = trim_batch(target_ids, tgt_pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, src_pad_token_id, attention_mask=masks)
        
        output_batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
            "correct_answers_str": [x["correct_answers_str"] for x in batch],
            "partially_correct_answers_str": [x["partially_correct_answers_str"] for x in batch]
        }

        return output_batch
