import argparse
import csv
import json

import numpy as np
import spacy
from datasets import load_metric
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import torch
from sari import *

def get_pred(model, tokenizer, text):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dummy_input = tokenizer(text, return_tensors='pt')
    dummy_input.to(device)
    output_tokens = model.generate(**dummy_input, num_beams=8,max_length=1024)
    output_sentence = tokenizer.batch_decode(output_tokens)
    output_sentence[0]=output_sentence[0].replace("</s>","")
    output_sentence[0]=output_sentence[0].replace("<s>","")
    return output_sentence


def main(args):
    # Load Model Artifacts
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    special_tokens_dict = {'additional_special_tokens': ['<clarity>', '<fluency>', '<coherence>', '<style>', '<meaning-changed>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    # Load References
    sources = []
    refs = []
    sources_sari = []
    with open(args.reference, "r") as f:
        for line in f:
            labels = json.loads(line)["labels"]
            if labels != "others":
                sources_sari.append(json.loads(line)["before_sent"])
                sources.append(json.loads(line)["before_sent_with_intent"])
                refs.append(json.loads(line)["after_sent"])

    preds=[]
    # Get Predictions
    for ins in tqdm(sources):
        preds.append(get_pred(model, tokenizer, ins)[0])

    metric_rouge = load_metric("rouge")
    metric_bleu = load_metric("bleu")

    # Compute Metrics
    # BLEU
    nlp = spacy.load("en_core_web_sm")
    pred_corpus = [[x.text for x in nlp(p)] for p in tqdm(preds)]
    ref_corpus = [[[x.text for x in nlp(p)]] for p in tqdm(refs)]
    assert len(pred_corpus) == len(ref_corpus)
    bleu_result = metric_bleu.compute(predictions=pred_corpus, references=ref_corpus)
    print('BLEU     :', bleu_result['bleu'])

    # ROUGE
    rouge_scores = metric_rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    # Extract a few results from ROUGE
    rouge_result = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
    print('ROUGE     :', rouge_result)

    # SARI
    sari_refs = [[ref] for ref in refs]

    sari_score, keep, delete, add = compute(sources=sources_sari, predictions=preds, references=sari_refs)
    print(f"SARI: {sari_score}, KEEP: {keep}, ADD: {add}, DELETE: {delete}")

    # Write to file if specified
    if args.output:
        print("inside output")

        with open(args.output + "_predictions_test_without_others.json", "w", newline='') as f:
            for ins, ref, pred in zip(sources, refs, preds):
                f.write(json.dumps({
                    "before_sent": ins,
                    "after_sent": ref,
                    "after_sent_gen": pred
                }) + "\n")

        with open(args.output + "_scores_test_without_others.json", "w") as f:
            f.write(json.dumps({
                "metrics": {
                    "BLEU": bleu_result['bleu'],
                    "ROUGE": rouge_result,
                    "SARI": sari_score,
                    "KEEP": keep,
                    "ADD": add,
                    "DELETE": delete
                }
            }))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', '-r', required=True,
                        help='path to reference sentences')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='path to BART model checkpoint')
    parser.add_argument('--output', '-o', required=False, default=None,
                        help='if specified all predictions and scores are '
                             'saved to the given path')

    args = parser.parse_args()
    main(args)
