# Style Classifier

This directory contains code to retrain the style classifier for the 11-way CDS styles using RoBERTa Large via `train_multilabel.py`.

To retrain the model, run:

```
python3 -m style_classifier.train_multilabel.py \ 
--use_accuracy_for_training \
--lr 5e-5 \
--batch_size 128 \
--seed 0
```

The `--use_accuracy_for_training` flag will display the classification accuracy on the dev set while training and use this to indicate the best checkpoint. Note this code contains other helpful stuff like early stopping (after 5 eval steps without improvement, etc.). Please look into the code/arguments for full details.

Note that we can also evaluate a trained model on the dev dataset as well using this code by using the `--evaluate` flag. The full command might look like:

```
python3 -m style_classifier.train_multilabel.py \ 
--pretrained_path $YOUR_TRAINED_MODEL_PATH \
--batch_size 128 \
--seed 0 \
--evaluate 
```

We upload the trained classifier from our model at https://huggingface.co/hallisky/cds_style_classifier.