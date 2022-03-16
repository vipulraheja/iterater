import argparse
import os
import re
import csv
from tqdm import tqdm
import json
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from scipy.special import expit
import numpy as np
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["WANDB_DISABLED"] = "true"

ctrl_tokens_dict = {}
ctrl_tokens_dict["clarity"] = 0
ctrl_tokens_dict["fluency"] = 1
ctrl_tokens_dict["coherence"] = 2
ctrl_tokens_dict["style"] = 3
ctrl_tokens_dict["meaning-changed"] = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    upsample_values = args.upsample_values # for oversampling
    epochs = args.epochs
    weights = args.weights
    if upsample_values is None:
        upsample_values = [1,1,1,1,1] # for oversampling
    if weights is None:
        weights = [1.,1.,1.,1.,1.] # for weighted loss
    pos_weight = torch.as_tensor(np.array(weights), dtype=torch.float, device=device)
    saving_name = args.saving_name
    model_name = 'roberta-large'
    tokenizer_cache_dir= 'roberta-large-tokenizer-cache/'
    model_cache_dir= 'roberta-large-model-cache/'
    model_type = RobertaModel
    config_type = RobertaConfig
    tokenizer_type = RobertaTokenizer

    dataset = load_dataset(args.dataset)
    train_before_sents, train_after_sents, train_labels = get_data(dataset['train'], upsample_values)
    test_before_sents, test_after_sents, test_labels =  get_data(dataset['validation'], None)
    tokenizer = tokenizer_type.from_pretrained(
        model_name,
        cache_dir=tokenizer_cache_dir
    )

    train_encodings = tokenizer(train_before_sents, train_after_sents, truncation=True, padding=True)
    test_encodings = tokenizer(test_before_sents, test_after_sents, truncation=True, padding=True)

    train_dataset = IntentDetectionDataset(train_encodings, train_labels)
    val_dataset = IntentDetectionDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=saving_name+'/',          # output directory
        num_train_epochs=epochs,              # total # of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=saving_name,            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=250,
        learning_rate=1e-5,
        gradient_accumulation_steps=4,
        save_steps=250,
        seed=171
    )

    config = config_type.from_pretrained(model_name)
    config.num_labels = 5
    config.classifier_dropout = config.hidden_dropout_prob

    model = CustomModel(training_args, model_type, model_name, config=config, cache_dir=model_cache_dir, pos_weight=pos_weight)
    model.to(device)


    trainer = Trainer(
        model=model,                  # the instantiated 🤗 Transformers model to be trained
        args=training_args,           # training arguments, defined above
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset     # evaluation dataset
    )

    trainer.train()
    print(trainer.evaluate())

def get_data(data, upsample_values):
    before_sents = []
    after_sents = []
    labels = []
    data_count = [0,0,0,0,0]
    label = []
    for line in data:
        before_sent = line["before_sent"]
        after_sent = line["after_sent"]
        label_str = line["labels"]
        if label_str == "others":
            continue
        label = ctrl_tokens_dict[label_str]
        upsample_value=1
        if upsample_values:
            upsample_value = upsample_values[label]
        for i in range(upsample_value):
            before_sents.append(before_sent)
            after_sents.append(after_sent)
            labels.append(label)
            data_count[label]+=1
    print(data_count)
    return before_sents, after_sents, labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'P_Clarity': precision[0],
        'R_Clarity': recall[0],
        'f1_Clarity': f1[0],
        'P_Fluency': precision[1],
        'R_Fluency': recall[1],
        'f1_Fluency': f1[1],
        'P_Coherence': precision[2],
        'R_Coherence': recall[2],
        'f1_Coherence': f1[2],
        'P_Style': precision[3],
        'R_Style': recall[3],
        'f1_Style': f1[3],
        'P_Meaning-Changed': precision[4],
        'R_Meaning-Changed': recall[4],
        'f1_Meaning-Changed': f1[4]
    }



class IntentDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomModel(nn.Module):
    def __init__(self, args, model_class, model_name_or_path, from_tf=False, config=None, cache_dir=None, pos_weight=None):
        super(CustomModel, self).__init__()
        self.config = config
        self.roberta = model_class.from_pretrained(model_name_or_path, from_tf=from_tf, config=config, cache_dir=cache_dir)
        self.num_labels = config.num_labels
        self.classifier = RobertaClassificationHead(config)
        self.pos_weight=pos_weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss(weight=self.pos_weight)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saving_name', '-s', required=True,
                        help='model name for saving')
    parser.add_argument('--dataset', '-d', required=True,
                        help='dataset for training and evaluation')
    parser.add_argument('--epochs', '-e', type=int, required=True,
                        help='number of epochs')
    parser.add_argument("--upsample_values", nargs=5, metavar=('a', 'b', 'c', 'd', 'e'),
                        help="upsampling ratio for labels", type=int,
                        default=None)
    parser.add_argument("--weights", nargs=5, metavar=('a', 'b', 'c', 'd', 'e'),
                        help="weights for labels for weighted loss", type=float,
                        default=None)

    args = parser.parse_args()
    main(args)

