## Train Pegasus and BART models

Run `train_pegasus.sh` and `train_pegasus.sh` respectively for training the pegasus and bart models respectively.

## Generate predictions and calculate metrics

To generate the sentence level predictions and for calculating metrics, use the following commands.

```
python pegasus_inference_and_metrics.py --checkpoint MODEL_CHECKPOINT --reference DATASET --output OUTPUT_LOCATION
```

```
python bart_inference_and_metrics.py --checkpoint MODEL_CHECKPOINT --reference DATASET --output OUTPUT_LOCATION
```
