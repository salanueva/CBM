import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertLayer, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform, BertConfig
import torch.nn.functional as F

from copy import deepcopy

# Original code from https://github.com/uclanlp/visualbert by Liunian Harold Li


class BertEmbeddingsWithVisualEmbedding(nn.Module):
    """Construct the embeddings from word, position, token_type embeddings and visual embeddings.
    """

    def __init__(self, config, visual_embedding_dim=2048):
        super(BertEmbeddingsWithVisualEmbedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Below are specific for encoding visual features

        # Segment and position embedding for image features
        self.token_type_embeddings_visual = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings_visual = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.projection = nn.Linear(visual_embedding_dim, config.hidden_size)
        
        # self.special_intialize()
        
    def special_intialize(self, method_type=0):
        # This is a bit unorthodox. The better way might be to add an initializer to AllenNLP. This function is used to
        # initialize the token_type_embeddings_visual and position_embedding_visual, just in case.
        self.token_type_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.token_type_embeddings.weight.data),
                                                                      requires_grad=True)
        self.position_embeddings_visual.weight = torch.nn.Parameter(deepcopy(self.position_embeddings.weight.data),
                                                                    requires_grad=True)
        return

    def forward(self, input_ids, token_type_ids=None, visual_embeddings=None, visual_embeddings_type=None):
        """
        input_ids = [batch_size, sequence_length]
        token_type_ids = [batch_size, sequence_length]
        visual_embeddings = [batch_size, image_feature_length, image_feature_dim]
        visual_embeddings_type = [batch_size, image_feature_length]
        """

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if visual_embeddings is not None:
            visual_embeddings = self.projection(visual_embeddings)
            token_type_embeddings_visual = self.token_type_embeddings_visual(visual_embeddings_type)
            v_embeddings = visual_embeddings + token_type_embeddings_visual

            # Concatenate the two:
            embeddings = torch.cat((embeddings, v_embeddings), dim=1)  # Concat visual embeddings after the attentions

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertVisualModel(BertModel):
    def __init__(self, config):
        super(BertVisualModel, self).__init__(config)
        self.embeddings = BertEmbeddingsWithVisualEmbedding(config, visual_embedding_dim=2048)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.bypass_transformer = False

        if self.bypass_transformer:
            self.additional_layer = BertLayer(config)

        self.output_attentions = config.output_attentions

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, input_ids, token_type_ids, attention_mask, visual_embeddings,
                visual_embeddings_type, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, visual_embeddings=visual_embeddings,
                                           visual_embeddings_type=visual_embeddings_type)

        if self.bypass_transformer and visual_embeddings is not None:
            assert (not output_all_encoded_layers)  # Don't support this for the bypass model
            text_length = input_ids.size(1)
            text_embedding_output = embedding_output[:, :text_length, :]
            visual_part = embedding_output[:, text_length:, :]

            text_extended_attention_mask = extended_attention_mask[:, :, :text_length, :text_length]

            encoded_layers = self.encoder(text_embedding_output,
                                          text_extended_attention_mask,
                                          output_hidden_states=output_all_encoded_layers)
            sequence_output = encoded_layers[0]
            new_input = torch.cat((sequence_output, visual_part), dim=1)
            final_sequence_output = self.additional_layer(new_input, extended_attention_mask)
            pooled_output = self.pooler(final_sequence_output)
            return final_sequence_output, pooled_output

        if self.output_attentions:
            encoded_layers, attn_data_list = self.encoder(embedding_output,
                                                          extended_attention_mask,
                                                          output_hidden_states=output_all_encoded_layers)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output, attn_data_list
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_hidden_states=output_all_encoded_layers)
            sequence_output = encoded_layers[0]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output



class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config_type, num_labels):
        config = BertConfig.from_pretrained(config_type)
        super().__init__(config)

        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(config_type)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.num_labels),
        )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
  
        return logits



class MMBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config_type, num_labels):
        config = BertConfig.from_pretrained(config_type)
        super().__init__(config)

        self.num_labels = num_labels

        self.bert = BertVisualModel.from_pretrained(config_type)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            BertPredictionHeadTransform(self.bert.config),
            nn.Linear(self.bert.config.hidden_size, self.num_labels),
        )
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        visual_embeddings=None,
        visual_embeddings_type=None
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
  
        return logits
