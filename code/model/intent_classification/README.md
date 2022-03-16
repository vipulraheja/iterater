This folder contains the script to train the Roberta-based classifier that predicts the edit intent given the source and target sentence pairs.

The model can be trained by using the following command. There are some parameters to specify.

`dataset`: the dataset name from huggingface datasets.

`upsample_values`: if upsampling in training data is required for the intents. By default, no upsampling is done. 

`weights`: for computing weighted loss for different intents. By default, all intents are equally weighted.

`epochs`: number of epochs for training the model.

`saving_name`: the name for the trained model.

```
python train_intent_classifier.py --epochs 15 --upsample_values 1 1 1 1 1 --weights 1. 1. 1. 1. 1. --saving_name roberta-large-test --dataset wanyu/IteraTeR_human_sent
```
