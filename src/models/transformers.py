
from transformers import BertTokenizer, logging

from models import models


def load_model(model_name, num_labels):
    
    # Set a higher level logging verbosity to clean the output text during the loading model process
    logging_level = logging.get_verbosity()
    logging.set_verbosity(logging.ERROR)

    # Load model initialized with the original pretrained weights
    if model_name in ["bertq", "cbm"]:
        model = models.BertForSequenceClassification(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=num_labels
        )
    elif model_name in ["mmbert", "cbm+mmbert"]:
        model = models.MMBertForSequenceClassification(
            config_type='bert-base-uncased',
            num_labels=num_labels
        )
    else:
        raise NotImplementedError
    
    # Come back to the original logging verbosity
    logging.set_verbosity(logging_level)
    
    return model


def load_tokenizer(model_name):
    if model_name in ["bertq", "cbm", "mmbert", "cbm+mmbert"]:
        return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    else:
        raise NotImplementedError