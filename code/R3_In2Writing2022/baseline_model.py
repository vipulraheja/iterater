import json
import os
import random
from copy import deepcopy
from difflib import SequenceMatcher
from typing import Dict

import torch
from sentsplit.config import en_config
from sentsplit.segment import SentSplit
from torch import nn
from transformers import (PegasusForConditionalGeneration, PegasusTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput


def load_sentsplit() -> SentSplit:
    my_config = deepcopy(en_config)
    my_config['maxcut'] = 800
    return SentSplit('en', **my_config)


random.seed(42)

class CustomModel(nn.Module):
    def __init__(self, args, model_class, model_name_or_path, from_tf=False, config=None, cache_dir=None, pos_weight=None):
        super(CustomModel, self).__init__()
        self.config = config
        self.bert = model_class.from_pretrained(model_name_or_path, from_tf=from_tf, config=config, cache_dir=cache_dir)
        self.num_labels = config.num_labels
        classifier_dropout = (
            config.classifier_dropout if (config.classifier_dropout is not None) else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
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
        outputs = self.bert(
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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    print("regression")
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    print("Single label classification")
                    self.config.problem_type = "single_label_classification"
                else:
                    print("multi label classification")
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
                #loss_fct = nn.BCEWithLogitsLoss()
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


class BaselineModel:
    def __init__(self, revision_model_path: str = '../models/2021-01-06-multi-intent',
                 binary_classifier_model_path: str = '../models/binary_intent_classifier',
                 intent_classifier_model_path: str = '../models/intent_classifier_src'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load text revision model
        self.model = PegasusForConditionalGeneration.from_pretrained(revision_model_path)
        self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
        self.model.to(self.device)
        self.model.eval()

        # tokenizer for both binary and intent classifier
        self.binary_and_intent_classifier_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', cache_dir='roberta-base-tokenizer-cache')

        # binary classifier
        config = RobertaConfig.from_pretrained(binary_classifier_model_path)
        self.binary_classifier_model = CustomModel(None, RobertaModel, binary_classifier_model_path, config=config)
        self.binary_classifier_model.load_state_dict(torch.load(os.path.join(binary_classifier_model_path, 'pytorch_model.bin')))
        self.binary_classifier_model.eval()
        self.binary_classifier_model.to(self.device)

        # intent classifier
        config = RobertaConfig.from_pretrained(intent_classifier_model_path)
        self.intent_classifier_model = CustomModel(None, RobertaModel, intent_classifier_model_path, config=config)
        self.intent_classifier_model.load_state_dict(torch.load(os.path.join(intent_classifier_model_path, 'pytorch_model.bin')))
        self.intent_classifier_model.eval()
        self.intent_classifier_model.to(self.device)

        # load sentence splitter
        self.splitter = load_sentsplit()

        self.tag_to_type = {'delete': 'D', 'replace': 'R', 'insert': 'A', 'equal': ''}
        if 'multi-intent' in revision_model_path:
            self.input_rep = 'intent_multi_sents'
            depth_tokens = [f'<{i}>' for i in range(1, 11)]
            special_tokens_dict = {'additional_special_tokens': ['<clarity>', '<fluency>', '<coherence>', '<style>', '<S>', '</S>'] + depth_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        elif 'single-intent' in revision_model_path:
            self.input_rep = 'intent_single_sent'
            depth_tokens = [f'<{i}>' for i in range(1, 11)]
            special_tokens_dict = {'additional_special_tokens': ['<clarity>', '<fluency>', '<coherence>', '<style>', '<S>', '</S>'] + depth_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        else:
            self.input_rep = 'single_sent'

    def predict_intent(self, before_sent) -> str:
        '''
        text_batch is a list of sentences
        tokenizer is bert's tokenizer
        model is the bert model
        '''
        input_ids = self.binary_and_intent_classifier_tokenizer(before_sent, return_tensors='pt', padding=True, truncation=True)
        rev_ctrl_tokens_dict = {0: 'clarity', 1: 'fluency', 2: 'coherence', 3: 'style'}

        with torch.no_grad():
            # binary intent classification
            input_ids = input_ids.to(self.device)
            binary_intent_outputs = self.binary_classifier_model(**input_ids)
            binary_intent_softmax_scores = torch.softmax(binary_intent_outputs.logits, dim=1).cpu().numpy()
            binary_intents = []
            for i in range(len(binary_intent_softmax_scores)):
                if binary_intent_softmax_scores[i][0] > binary_intent_softmax_scores[i][1]:
                    binary_intents.append('no_intent')
                else:
                    binary_intents.append('yes_intent')

            # n-ary intent classification
            intent_outputs = self.intent_classifier_model(**input_ids)
            intent_sigmoid_scores = torch.sigmoid(intent_outputs.logits)
            mask = torch.ones(intent_sigmoid_scores.size())
            mask_zero = torch.zeros(intent_sigmoid_scores.size())
            intent_ans = (torch.where(intent_sigmoid_scores.cpu() >= 0.3, mask, mask_zero)).numpy()
            intents = []
            for i in range(len(intent_ans)):
                intent = []
                for j in range(len(intent_ans[i])):
                    if intent_ans[i][j] == 1:
                        intent.append(rev_ctrl_tokens_dict[j])
                intents.append(intent)

            # print(binary_intents)
            # print(intents)
            assert len(binary_intents) == len(intents)
            final_intents = []
            for binary, intent in zip(binary_intents, intents):
                if binary == 'no_intent':
                    final_intents.append(['no_intent'])
                else:
                    final_intents.append(intent)
            assert len(final_intents) == 1
            # print(final_intents)
            return final_intents[0][0]

    def revise(self, document: Dict) -> Dict:
        '''
        Given a document (Dict), model adds `after_revision` and `edit_actions`:
            {
                "doc_id": "65581384",
                "before_revision": "before document",
                "after_revision": "revised document",
                "edit_actions": [
                    {
                        "type": "R",
                        "before": "has",
                        "after": "had",
                        "start_char_pos": 214,
                        "end_char_pos": 217,
                        "major_intent": "fluency",
                    },
                    ..
                ]
            }
        '''
        intents = ['clarity', 'fluency', 'coherence', 'style']

        before_sents = self.splitter.segment(document['before_revision'])
        after_sents, edits = [], []
        cuml_char_index = 0
        for sent_id, before_sent in enumerate(before_sents):
            major_intent = self.predict_intent(before_sent)

            if major_intent == 'no_intent':
                major_intent = random.sample(intents, 1)[0]

            if self.input_rep == 'intent_single_sent':
                before_sent_with_intent = f'<{major_intent}> ' + before_sent
            elif self.input_rep == 'intent_multi_sents':
                if sent_id - 2 >= 0:
                    ctx_prev = ' '.join(before_sents[sent_id-2: sent_id])
                else:
                    if sent_id-1 >= 0:
                        ctx_prev = ' '.join(before_sents[sent_id-1: sent_id])
                    else:
                        ctx_prev = ''
                ctx_after = ' '.join(before_sents[sent_id+1: sent_id+3])
                before_sent_with_intent = f'<{major_intent}> {ctx_prev.strip()} <S> {before_sent.strip()} </S> {ctx_after.strip()}'
            else:
                before_sent_with_intent = before_sent

            before_input = self.tokenizer(before_sent_with_intent, return_tensors='pt').to(self.device)
            output = self.model.generate(**before_input)
            after_sent = self.tokenizer.batch_decode(output)[0]

            after_sents.append(after_sent)

            s = SequenceMatcher(None, before_sent, after_sent)
            for tag, i1, i2, j1, j2 in s.get_opcodes():
                edit_type = self.tag_to_type[tag]
                if len(edit_type) > 0:
                    if before_sent[i1:i2] == ' ': continue
                    edits.append({
                            "type": edit_type,
                            "before": before_sent[i1:i2],
                            "after": after_sent[j1:j2],
                            "start_char_pos": i1 + cuml_char_index,
                            "end_char_pos": i2 + cuml_char_index,
                            "major_intent": major_intent,
                            })
            cuml_char_index += len(before_sent)

        model_rev = {
                "doc_id": document['doc_id'],
                "before_revision": document['before_revision'],
                "after_revision": ' '.join(after_sents),
                "edit_actions": edits
                }
        return model_rev


if __name__ == '__main__':
    model = BaselineModel(revision_model_path='google/pegasus-large')

    with open('dummy_test_data.json', 'r') as f:
        original_test_data = json.load(f)
    for cur_doc in original_test_data:
        if cur_doc['doc_id'] == '51157723':
            model_revised_doc = model.revise(cur_doc)
            print(model_revised_doc)