import argparse
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils import data
import transformers
import shutil


from data.okvqa_seq2seq import Seq2SeqOkVqaDataset
from data.constants import *
from utils import utils_t5


class LitModel(pl.LightningModule):
    
    def __init__(self, args):
        super().__init__()
        
        # Load model, tokenizer and loss function
        self.model_name = args.model
        self.model = transformers.T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.model_name)

        # Define other hyperparameters
        self.batch_size = args.batch_size
        self.warmup_steps = args.warmup_steps # 0
        self.max_steps = args.max_steps # 50 epochs
        self.lr = args.lr
        self.opt_eps = args.opt_eps 
        self.opt_wd = args.opt_wd
    
        self.fast_version = args.fast_version
        self.deepspeed = args.deepspeed
        self.max_target_length = args.max_target_length
        self.train_val_splitted = args.train_val_splitted

        self.predictions = []

        self.q_ids = []
        if args.q_ids_path is not None:
            with open(args.q_ids_path, "r") as f:
                self.q_ids = json.load(f)
        

    def full_step(self, batch, split="train", batch_idx=0):
        
        self.generate_step(batch, split=split, batch_idx=batch_idx)
        return self.step(batch, split=split)
    

    def step(self, batch, split="train"):
        
        batch_size = len(batch["correct_answers_str"])
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["decoder_input_ids"])
        loss = outputs.loss

        self.log(f'{split}_loss', loss.detach(), on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=batch_size)

        return loss   

    def generate_step(self, batch, split="test", batch_idx=0):
        
        batch_size = len(batch["correct_answers_str"])
        gen_outputs = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        gen_text = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        if split == "test":
            current_q_ids = self.q_ids[(batch_idx * self.batch_size):(batch_idx * self.batch_size + batch_size)]
            self.predictions += [{"question_id": idx, "answer": ans} for idx, ans in zip(current_q_ids, gen_text)]

        correct_answers_str = batch["correct_answers_str"]
        partially_correct_answers_str = batch["partially_correct_answers_str"] 
        accuracy = utils_t5.compute_okvqa_accuracy(gen_text, correct_answers_str, partially_correct_answers_str)
        
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=(split=="train"), logger=True, batch_size=batch_size)

        return accuracy

    def configure_optimizers(self):
        # Define optimizer parameter groups
        
        if self.deepspeed:
            optimizer = DeepSpeedCPUAdam(self.model.parameters(), lr=self.lr, eps=self.opt_eps, weight_decay=self.opt_wd,)
        else:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.opt_wd,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0
                },
            ]
            optimizer = transformers.AdamW(optimizer_parameters, lr=self.lr, eps=self.opt_eps)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        if self.fast_version:
            return self.step(batch, split="train")
        else:
            return self.full_step(batch, split="train")
    
    def validation_step(self, batch, batch_idx):
        if self.fast_version:
            return self.generate_step(batch, split="val")
        else:
            if self.train_val_splitted:
                return self.full_step(batch, split="val")
            else:
                return self.generate_step(batch, split="val")

    def test_step(self, batch, batch_idx):
        if self.fast_version:
            return self.generate_step(batch, split="test", batch_idx=batch_idx)
        else:
            return self.full_step(batch, split="test", batch_idx=batch_idx)
    

class OkvqaDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, args):
        super().__init__()

        self.tokenizer = tokenizer

        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

        self.n_obs_train = args.n_obs_train
        self.n_obs_val = args.n_obs_val
        self.n_obs_test = args.n_obs_test

        self.train_val_splitted = args.train_val_splitted

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def train_dataloader(self):

        dataset = Seq2SeqOkVqaDataset(
            tokenizer=self.tokenizer, 
            root=DATA_ROOT,
            q_file=OKVQA_TRAIN_QUESTION_FILE,
            ann_file=OKVQA_TRAIN_ANN_FILE_V1_1,
            cap_type="oscar",
            max_source_length=self.max_source_length, 
            max_target_length=self.max_target_length, 
            split="train", 
            n_obs=self.n_obs_train, 
            train_val_splitted=self.train_val_splitted
        )
        
        params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }

        return data.DataLoader(dataset, **params)
    
    def val_dataloader(self):
        
        q_file = OKVQA_TRAIN_QUESTION_FILE if self.train_val_splitted else OKVQA_TEST_QUESTION_FILE
        ann_file = OKVQA_TRAIN_ANN_FILE_V1_1 if self.train_val_splitted else OKVQA_TEST_ANN_FILE_V1_1

        dataset = Seq2SeqOkVqaDataset(
            tokenizer=self.tokenizer, 
            root=DATA_ROOT,
            q_file=q_file,
            ann_file=ann_file,
            cap_type="oscar",
            max_source_length=self.max_source_length, 
            max_target_length=self.max_target_length, 
            split="val", 
            n_obs=self.n_obs_val, 
            train_val_splitted=self.train_val_splitted
        )
        
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }

        return data.DataLoader(dataset, **params)
    
    def test_dataloader(self):
        
        dataset = Seq2SeqOkVqaDataset(
            tokenizer=self.tokenizer, 
            root=DATA_ROOT,
            q_file=OKVQA_TEST_QUESTION_FILE,
            ann_file=OKVQA_TEST_ANN_FILE_V1_1,
            cap_type="oscar",
            max_source_length=self.max_source_length, 
            max_target_length=self.max_target_length, 
            split="test", 
            n_obs=self.n_obs_test, 
            train_val_splitted=self.train_val_splitted
        )
        
        params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'collate_fn': dataset.collate_fn
        }

        return data.DataLoader(dataset, **params)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="t5-base", choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before training."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--output_path", type=str, default="./output", help="Output directory for plots and models."
    )
    parser.add_argument(
        "--q_ids_path", type=str, default=None, help="Json file with ordered list of question ids."
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
        "--warmup_steps", type=int, default=0, help="Warmup steps to be done during training."
    )
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="Steps to be done during training."
    )

    parser.add_argument(
        "--n_obs_train", type=int, default=None, help="Number of training instances."
    )
    parser.add_argument(
        "--n_obs_val", type=int, default=None, help="Number of validation instances."
    )
    parser.add_argument(
        "--n_obs_test", type=int, default=None, help="Number of test instances."
    )
    parser.add_argument(
        "--max_source_length", type=int, default=256, help="Max character length of source text."
    )
    parser.add_argument(
        "--max_target_length", type=int, default=128, help="Max character length of target text."
    )

    parser.add_argument(
        "--num_workers", type=int, default=8, help="Workers used in the dataloader."
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="Seed."
    )

    parser.add_argument(
        "--do_train", action="store_true", help="If true, trains model before evaluation."
    )
    parser.add_argument(
        "--do_predict", action="store_true", help="If true, test model after fine-tuning."
    )
    parser.add_argument(
        "--train_val_splitted", action="store_true", help="Use the splitted training version."
    )
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16 during training."
    )
    parser.add_argument(
        "--accumulate_grad_batches", type=int, default=1, help="Backpropagation will be applied over the accumulation of this number of batches."
    )
    parser.add_argument(
        "--deepspeed", action="store_true", help="If true, the model and gradients will be distributed in multiple GPUs and even RAM."
    )
    parser.add_argument(
        "--fast_version", action="store_true", help="If true, training and evaluation will be faster, but logs will be incomplete... (no accuracy computed in train, no loss computed in val)"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )

    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()
    
    # Reproducibility
    if args.seed != -1:
        pl.utilities.seed.seed_everything(args.seed, workers=True)

    # Load model
    print("Loading model...")
    if args.ckpt is None:
        model = LitModel(args)
    else: 
        model = LitModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=True)
    
    # Load data
    print("Loading dataset...")
    datamodule = OkvqaDataModule(model.tokenizer, args)
    
    # Define checkpoint filename and tensorboard run name
    tb_run_name = utils_t5.generate_run_name(args.model, args.output_path, args.run_name)

    # Define precision
    precision = 32
    if args.bf16:
        precision = "bf16"

    # Define deepspeed strategy
    deepspeed = None
    if args.deepspeed and args.gpus > 1:
        deepspeed = DeepSpeedPlugin(offload_optimizer=True, allgather_bucket_size=5e8, reduce_bucket_size=5e8)
    
    # Define gradient accumulation steps
    accumulate_grad_batches = args.accumulate_grad_batches

    # Define logger and trainer
    logger = TensorBoardLogger("logs", name=tb_run_name, default_hp_metric=False)
    trainer = pl.Trainer(checkpoint_callback=False, gpus=args.gpus, precision=precision, accumulate_grad_batches=accumulate_grad_batches, strategy=deepspeed, logger=logger, max_steps=args.max_steps)
    
    # Train model
    if args.do_train:
        print("Training starts!")
        trainer.fit(model, datamodule)
        print("Training finished!")
        if deepspeed is not None:
            trainer.save_checkpoint(os.path.join(args.output_path, tb_run_name), weights_only=True)    
            convert_zero_checkpoint_to_fp32_state_dict(os.path.join(args.output_path, tb_run_name), os.path.join(args.output_path, tb_run_name + ".pth"))
        else:
            trainer.save_checkpoint(os.path.join(args.output_path, tb_run_name + ".pth"), weights_only=True)

    # Evaluate model
    if args.do_predict:
        trainer.test(model, dataloaders=datamodule.test_dataloader())
        with open(os.path.join(args.output_path, utils_t5.generate_prediction_filename(args.model, args.output_path, tb_run_name)), "w") as f:
            json.dump(model.predictions, f)
    
    # Delete unneccesary copies of the model
    if args.deepspeed:
        shutil.rmtree(os.path.join(args.output_path, tb_run_name))



if __name__ == '__main__':
    main()
