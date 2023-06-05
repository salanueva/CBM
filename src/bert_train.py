import argparse
import json
from typing import Any, Dict
import numpy as np
import os
import torch
from torch import optim
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import transformers


from data.constants import *
from data.okvqa import OkVqa
from data.vqa import Vqa
from models import losses
import models.transformers as models
from utils import utils_bert



class LitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        # Define input data of the model
        self.input_data = "q" # question
        if "cbm" in args.model:
            self.input_data += "c" # caption
        if "mmbert" in args.model:
            self.input_data += "f" # features (of images)

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
        self.model_name = args.model # bertq / cbm / mmbert / cbm+mmbert
        self.model = models.load_model(args.model, self.num_labels)
        self.tokenizer = models.load_tokenizer(args.model)
        self.loss = losses.SoftCrossEntropyLoss()

        # Define other hyperparameters
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps
        self.lr = args.lr
        self.opt_eps = args.opt_eps
        self.opt_wd = args.opt_wd

        self.pretrained_on = None
        self.prev_num_labels = 0
        

    def forward(self, batch):
        
        features, caption, question, _ = batch

        # Text embeddings
        if "c" in self.input_data:
            tokens = self.tokenizer(caption, question, padding='longest', max_length = 512, return_tensors='pt')
            # BUG: If I don't check this, the backward pass will fail, not sure why...
            if len(tokens.input_ids[0]) < 17:        
                tokens = self.tokenizer(caption, question, padding='max_length', max_length = 17, return_tensors='pt')
        else:
            tokens = self.tokenizer(question, padding='longest', max_length = 512, return_tensors='pt')
            # BUG: If I don't check this, the backward pass will fail, not sure why...
            if len(tokens.input_ids[0]) < 17:        
                tokens = self.tokenizer(question, padding='max_length', max_length = 17, return_tensors='pt')
        
        if "f" in self.input_data: # mmbert / cbm+mmbert
            # Visual features and attention mask
            visual_embeddings_type = torch.zeros(size=(features.size(0), features.size(1)), dtype=torch.long).to(self.device)
            input_ids = tokens.input_ids.to(self.device)
            token_type_ids = tokens.token_type_ids.to(self.device)
            att_mask = torch.tensor(
                [
                    [int(token_id > 0) for token_id in inst_ids] + [1] * int(feat.size(0))
                    for inst_ids, feat in zip(input_ids, features)
                ]
            ).to(self.device)
            
            # Forward pass
            logits = self.model(
                input_ids=input_ids, # [bs, seq_len]
                token_type_ids=token_type_ids, # [bs, seq_len]
                attention_mask=att_mask, # [bs, seq_len + feat_len]
                visual_embeddings=features,  # [bs, feat_len, feat_size]
                visual_embeddings_type=visual_embeddings_type  # [bs, feat_len]
            )

        else: # bertq / cbm
        
            # Forward pass
            logits = self.model(**tokens.to(self.device))
        
        return logits

    def configure_optimizers(self):
        # Define optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9), eps=self.opt_eps, weight_decay=self.opt_wd)
        scheduler = {
            "scheduler": transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.max_steps),
            "interval": "step"
        }
        return [optimizer], [scheduler]
    
    def general_step(self, batch, split="train"):
        
        # Model forward pass
        logits = self(batch)

        # MOVE: Load target data 
        answers = batch[-1]
        transposed_answers = list(map(list, zip(*answers)))

        target = losses.build_target_tensor(transposed_answers, self.label_ids).to(self.device)

        loss = self.loss(logits, target)

        # Compute Accuracy (don't take UNK token into account)
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
        
        # Save metrics
        self.log(f'{split}_loss', loss, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=len(predictions))
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=len(predictions))

        return loss

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, split="test")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.prev_num_labels = checkpoint['state_dict']['model.classifier.1.weight'].size(0)
        
        if self.prev_num_labels == 2253:
            self.pretrained_on = "okvqa_v1.0"
        elif self.prev_num_labels == 2250:
            self.pretrained_on = "okvqa_v1.1"
        elif self.prev_num_labels == 3129:
            self.pretrained_on = "vqa_v2"
        else:
            self.pretrained_on = "other"
        
        # Remove classifier layer's state dict if it was pretrained in another vqa task
        if self.pretrained_on != self.task_name:
            del checkpoint['state_dict']['model.classifier.1.weight']
            del checkpoint['state_dict']['model.classifier.1.bias']
        
        return super().on_load_checkpoint(checkpoint)


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
        self.is_tiny =args.tiny

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):
        dataset = OkVqa(DATA_ROOT, OKVQA_TRAIN_QUESTION_FILE, self.train_ann_file, feat_path=self.feat_path,
                        cap_type=self.cap_type, ans_type="stemmed", is_tiny=self.is_tiny)
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers
        }

        return data.DataLoader(dataset, **params)
    
    def test_dataloader(self):
        dataset = OkVqa(DATA_ROOT, OKVQA_TEST_QUESTION_FILE, self.test_ann_file, feat_path=self.feat_path,
                        cap_type=self.cap_type, ans_type="stemmed", is_tiny=self.is_tiny)
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
        self.is_tiny =args.tiny

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
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        return data.DataLoader(dataset, **params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="cbm", choices=["bertq", "cbm", "mmbert", "cbm+mmbert"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--cap_type", type=str, default="oscar", choices=["oscar", "human"], help="Type of caption that will be used: 'oscar' or 'human'."
    )

    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--opt_eps", type=float, default=1e-8, help="Epsilon value for AdamW optimizer."
    )
    parser.add_argument(
        "--opt_wd", type=float, default=0.0, help="Weight decay value for AdamW optimizer."
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=2000, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=88000, help="Steps to be done during training."
    )

    parser.add_argument(
        "--dataset", type=str, default="okvqa_v1.1", choices=["okvqa_v1.0", "okvqa_v1.1", "vqa_v2"],
        help="Select dataset to be trained on."
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny version of model, for tiny purposes..."
    )
    parser.add_argument(
        "--num_workers", type=int, default=12, help="Workers used in the dataloader."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Test model after fine-tuning."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="Output directory for plots and models."
    )

    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed)

    # Load model
    print("Loading model...")
    if args.ckpt is None:
        model = LitModel(args)
    else: 
        model = LitModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=False)
    
    # Load data
    print("Loading dataset...")
    if "okvqa" in args.dataset:
        datamodule = OkvqaDataModule(args)
    else:
        datamodule = VqaDataModule(args)
    
    # Define checkpoint filename and tensorboard run name
    ckpt_filename = utils_bert.generate_ckpt_filename(args.model, args.cap_type, args.dataset, args.output_path, model.pretrained_on, args.run_name)
    tb_run_name = ckpt_filename[:-5]

    # Define trainer
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    trainer = pl.Trainer(checkpoint_callback=False, gpus=args.gpus, fast_dev_run=args.tiny, logger=logger, max_steps=args.max_steps)

    # Train model
    print("Training starts!")
    trainer.fit(model, datamodule)
    print("Training finished!")
    trainer.save_checkpoint(os.path.join(args.output_path, ckpt_filename))

    # Evaluate model
    if args.evaluate:
        trainer.test(model, datamodule.test_dataloader())


if __name__ == "__main__":
    main()
