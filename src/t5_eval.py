import argparse
import json
import pytorch_lightning as pl
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.utils import data
import transformers
import os

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
        self.fast_version = args.fast_version
        self.max_target_length = args.max_target_length

        self.predictions = []
        

    def full_step(self, batch, split="train"):
        
        self.generate_step(batch, split=split)
        return self.step(batch, split=split)
    

    def step(self, batch, split="train"):
        
        batch_size = len(batch["correct_answers_str"])
        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["decoder_input_ids"])
        loss = outputs.loss

        self.log(f'{split}_loss', loss.detach(), on_epoch=True, prog_bar=True, logger=False, batch_size=batch_size)

        return loss   

    def generate_step(self, batch, split="test"):
        
        batch_size = len(batch["correct_answers_str"])
        gen_outputs = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        gen_text = self.tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

        self.predictions += gen_text

        correct_answers_str = batch["correct_answers_str"]
        partially_correct_answers_str = batch["partially_correct_answers_str"] 
        accuracy = utils_t5.compute_okvqa_accuracy(gen_text, correct_answers_str, partially_correct_answers_str)
        
        self.log(f'{split}_accuracy', accuracy, on_epoch=True, prog_bar=True, logger=False, batch_size=batch_size)

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


    def test_step(self, batch, batch_idx):
        if self.fast_version:
            return self.generate_step(batch, split="test")
        else:
            return self.full_step(batch, split="test")
    

class OkvqaDataModule(pl.LightningDataModule):

    def __init__(self, tokenizer, args):
        super().__init__()

        self.tokenizer = tokenizer

        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

        self.n_obs_test = args.n_obs_test
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
    
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
        "--ckpt", type=str, default=None, help="Model's checkpoint to be loaded before prediction."
    )
    parser.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs in use. (0 == cpu)"
    )
    parser.add_argument(
        "--q_ids_path", type=str, help="Json file with input question_id list."
    )    
    parser.add_argument(
        "--output_path", type=str, default="./output", help="Output directory for plots and models."
    )
    parser.add_argument(
        "--output_filename", type=str, default="./output", help="Output json filename."
    )


    parser.add_argument(
        "--batch_size", type=int, default=56, help="Batch size (per gpu)."
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
        "--fast_version", action="store_true", help="If true, training and evaluation will be faster, but logs will be incomplete..."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Name of the run. Used in tensorboard and output filenames. If it is not filled or already exists, a custom one will be generated."
    )

    args = parser.parse_args()
    return args


def main():
    
    args = parse_args()

    # Load model
    print("Loading model...")
    model = LitModel.load_from_checkpoint(checkpoint_path=args.ckpt, args=args, strict=False)
    
    # Load data
    print("Loading dataset...")
    datamodule = OkvqaDataModule(model.tokenizer, args)
    trainer = pl.Trainer(gpus=args.gpus)

    # Evaluate model
    trainer.test(model, dataloaders=datamodule.test_dataloader())
    # output_filename = utils.generate_predictions_filename(args.model, args.output_path, args.run_name)

    with open(args.q_ids_path, "r") as f:
        q_ids = json.load(f)
    
    predictions_dict = [{"question_id": idx, "answer": ans} for idx, ans in zip(q_ids, model.predictions)]

    with open(os.path.join(args.output_path, args.output_filename), "w") as f:
        json.dump(predictions_dict, f)


if __name__ == '__main__':
    main()
