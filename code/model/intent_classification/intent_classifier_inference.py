import json
import torch
import argparse
import numpy as np
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import Trainer, TrainingArguments, RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification

def main(args):
    checkpoint = args.checkpoint
    model_name = 'roberta-large'
    model_cache_dir='roberta-large-model-cache/'
    model_type = RobertaForSequenceClassification
    config_type = RobertaConfig
    tokenizer_type = RobertaTokenizer
    tokenizer = tokenizer_type.from_pretrained(
            model_name,
            cache_dir=model_cache_dir
    )

    id2label = {0: "clarity", 1: "fluency", 2: "coherence", 3: "style", 4: "meaning-changed"}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    model = model_type.from_pretrained(checkpoint)
    model.eval()
    model.to(device)

    before_text = 'I likes coffee.'
    after_text = 'I like coffee.'

    def score_text(before_text, after_text, tokenizer, model):

        input_ids = tokenizer(before_text, after_text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            input_ids = input_ids.to(device)
            outputs = model(**input_ids)
            softmax_scores = torch.softmax(outputs.logits, dim=1)
            softmax_scores = softmax_scores[0].cpu().numpy()
            index = np.argmax(softmax_scores)
            return index, softmax_scores[index]

    index, confidence = score_text([before_text], [after_text], tokenizer, model)
    label = id2label[index]
    print(label)
    print(confidence)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='path to Pegasus model checkpoint')

    args = parser.parse_args()
    main(args)