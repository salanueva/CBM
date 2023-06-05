from copy import deepcopy

import argparse
import json
import numpy as np
import os
import torch
from torch import optim
from torch.utils import data
import transformers


from data.constants import *
from data.okvqa import OkVqa
from data.vqa import Vqa
from models import losses
import models.transformers as models
from utils import utils_bert

import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        # Define input data of the model
        self.input_data_cbm = "cq" # caption + question
        self.input_data_mmbert = "fq" # features + question

        # MOVE: Load task labels
        self.task_name = args.dataset
        data = self.task_name.split("_")
        self.dataset = data[0]
        self.dataset_version = data[1]
        
        with open(os.path.join(DATA_ROOT, "vocab", f"{self.dataset}_answer_vocab_idx2word_{self.dataset_version}.json"), "r") as f:
            self.labels = json.load(f)
            self.labels.append("UNK")
            self.label_ids = {l: i for i, l in enumerate(self.labels)}
            self.num_labels = len(self.labels)
        
        # Load model, tokenizer and loss function
        self.model_cbm = models.load_model("cbm", self.num_labels)
        self.model_mmbert = models.load_model("mmbert", self.num_labels)

        # Load state_dicts
        cbm_sd = torch.load(args.ckpt_cbm)['state_dict']
        mmbert_sd = torch.load(args.ckpt_mmbert)['state_dict']

        if cbm_sd['model.classifier.1.weight'].size(0) != mmbert_sd['model.classifier.1.weight'].size(0):
            raise ValueError(f"These models are not fine-tuned in the same dataset.")
        
        if cbm_sd['model.classifier.1.weight'].size(0) != self.num_labels:
            raise ValueError(f"This model is not fine-tuned in the {self.task_name} dataset.")

        for key in list(cbm_sd.keys()):
            cbm_sd[key.replace('model.', '')] = cbm_sd.pop(key)

        for key in list(mmbert_sd.keys()):
            mmbert_sd[key.replace('model.', '')] = mmbert_sd.pop(key)

        self.model_cbm.load_state_dict(cbm_sd)
        self.model_mmbert.load_state_dict(mmbert_sd)

        self.tokenizer = models.load_tokenizer("cbm+mmbert")
        self.loss = losses.SoftCrossEntropyLoss()

        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, batch):
        
        features, caption, question, _ = batch

        ### CBM Model
        cbm_tokens = self.tokenizer(caption, question, padding='longest', max_length = 512, return_tensors='pt')
        # BUG: If I don't check this, the forward pass will fail, not sure why...
        if len(cbm_tokens.input_ids[0]) < 17:        
            cbm_tokens = self.tokenizer(caption, question, padding='max_length', max_length = 17, return_tensors='pt')
        
        cbm_logits = self.model_cbm(**cbm_tokens.to(self.device))
        
        ### MMBERT Model
        mmbert_tokens = self.tokenizer(question, padding='longest', max_length = 512, return_tensors='pt')
        # BUG: If I don't check this, the forward pass will fail, not sure why...
        if len(mmbert_tokens.input_ids[0]) < 17:        
            mmbert_tokens = self.tokenizer(question, padding='max_length', max_length = 17, return_tensors='pt')

        # Visual features and attention mask
        input_ids = mmbert_tokens.input_ids.to(self.device)
        token_type_ids = mmbert_tokens.token_type_ids.to(self.device)
        att_mask = torch.tensor(
            [
                [int(token_id > 0) for token_id in inst_ids] + [1] * int(feat.size(0))
                for inst_ids, feat in zip(input_ids, features)
            ]
        ).to(self.device)

        visual_embeddings_type = torch.zeros(size=(features.size(0), features.size(1)), dtype=torch.long).to(self.device)
        
        # Forward pass
        mmbert_logits = self.model_mmbert(
            input_ids=input_ids, # [bs, seq_len]
            token_type_ids=token_type_ids, # [bs, seq_len]
            attention_mask=att_mask, # [bs, seq_len + feat_len]
            visual_embeddings=features,  # [bs, feat_len, feat_size]
            visual_embeddings_type=visual_embeddings_type  # [bs, feat_len]
        )

        return cbm_logits, mmbert_logits

    def configure_optimizers(self):
        # Define optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd)
        scheduler = {
            "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
            "interval": "step"
        }
        return [optimizer], [scheduler]
    
    def compute_accuracy(self, logits, transposed_answers):
        
        _, indices = logits.topk(2)
        predictions = [
            int(ids[0].data.cpu().numpy())
            if ids[0] < len(self.labels) - 1
            else int(ids[1].data.cpu().numpy())
            for ids in indices
        ]

        accuracy = np.mean(
            np.array(
                [utils_bert.vqa_accuracy(ans, self.labels[int(p_id)]) for ans, p_id in zip(transposed_answers, predictions)]
            )
        )
        return accuracy

    def general_step(self, batch):
        
        # Model forward pass
        cbm_logits, mmbert_logits = self(batch)
        logits = torch.mul(self.sigmoid(cbm_logits), self.sigmoid(mmbert_logits))

        # MOVE: Load target data 
        answers = batch[-1]
        transposed_answers = list(map(list, zip(*answers)))
        target = losses.build_target_tensor(transposed_answers, self.label_ids).to(self.device)

        cbm_loss = self.loss(cbm_logits, target)
        mmbert_loss = self.loss(mmbert_logits, target)
        fusion_loss = self.loss(logits, target)

        # Compute Accuracy
        cbm_acc = self.compute_accuracy(cbm_logits, transposed_answers)
        mmbert_acc = self.compute_accuracy(mmbert_logits, transposed_answers)
        fusion_acc = self.compute_accuracy(logits, transposed_answers)
        
        # Save metrics
        self.log('cbm_loss', cbm_loss, on_epoch=True, prog_bar=False, batch_size=len(transposed_answers))
        self.log('mmbert_loss', mmbert_loss, on_epoch=True, prog_bar=False, batch_size=len(transposed_answers))
        self.log('loss', fusion_loss, on_epoch=True, prog_bar=True, batch_size=len(transposed_answers))
        self.log('cbm_accuracy', cbm_acc, on_epoch=True, prog_bar=False, batch_size=len(transposed_answers))
        self.log('mmbert_accuracy', mmbert_acc, on_epoch=True, prog_bar=False, batch_size=len(transposed_answers))
        self.log('accuracy', fusion_acc, on_epoch=True, prog_bar=True, batch_size=len(transposed_answers))

        return fusion_loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch)
    
    def test_step(self, batch, batch_idx):
        return self.general_step(batch)


class OkvqaDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()

        data = args.dataset.split("_")
        self.dataset_version = data[1]

        if self.dataset_version == "v1.0":
            self.train_ann_file = OKVQA_TRAIN_ANN_FILE_V1_0
            self.test_ann_file = OKVQA_TEST_ANN_FILE_V1_0
        elif self.dataset_version == "v1.1":
            self.train_ann_file = OKVQA_TRAIN_ANN_FILE_V1_1
            self.test_ann_file = OKVQA_TEST_ANN_FILE_V1_1
        else:
            raise NotImplementedError
        
        self.feat_path = FEAT_FOLDER
        self.cap_type = args.cap_type

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):
        dataset = OkVqa(DATA_ROOT, OKVQA_TRAIN_QUESTION_FILE, self.train_ann_file, feat_path=self.feat_path,
                        cap_type=self.cap_type, ans_type="stemmed")
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers
        }

        return data.DataLoader(dataset, **params)
    
    def test_dataloader(self):
        dataset = OkVqa(DATA_ROOT, OKVQA_TEST_QUESTION_FILE, self.test_ann_file, feat_path=self.feat_path,
                        cap_type=self.cap_type, ans_type="stemmed")
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }

        return data.DataLoader(dataset, **params)


class VqaDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        
        self.feat_path = FEAT_FOLDER
        self.cap_type = args.cap_type

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):
        dataset =  Vqa(DATA_ROOT, VQA_TRAIN_QUESTION_FILE, ann_file=VQA_TRAIN_ANN_FILE, feat_path=self.feat_path, cap_type=self.cap_type)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers
        }
        return data.DataLoader(dataset, **params)
    
    def test_dataloader(self):
        dataset = Vqa(DATA_ROOT, VQA_VAL_QUESTION_FILE, ann_file=VQA_VAL_ANN_FILE, feat_path=self.feat_path, cap_type=self.cap_type)
        params = {
            'batch_size': 58,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        return data.DataLoader(dataset, **params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_cbm", type=str, default=None, help="CBM's checkpoint to be loaded."
    )
    parser.add_argument(
        "--ckpt_mmbert", type=str, default=None, help="MMBERT's checkpoint to be loaded."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--cap_type", type=str, default="oscar", choices=["oscar", "human"], help="Type of caption that will be used: 'oscar' or 'human'."
    )

    parser.add_argument(
        "--dataset", type=str, default="okvqa_v1.1", choices=["okvqa_v1.0", "okvqa_v1.1", "vqa_v2"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader."
    )

    args = parser.parse_args()
    return args



def main():
    
    args = parse_args()
    
    # Load model and data
    model = LitModel(args=args)
    if "okvqa" in args.dataset:
        datamodule = OkvqaDataModule(args)
    else:
        datamodule = VqaDataModule(args)
    
    # Define trainer
    trainer = pl.Trainer(checkpoint_callback=False, gpus=args.gpus)

    # Evaluate model
    trainer.test(model, datamodule.test_dataloader())

if __name__ == "__main__":
    main()