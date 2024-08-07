from model_utils_bert import BertRnn, BertRnnDist
from transformers import Trainer, TrainingArguments, BertConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline, BartForSequenceClassification, set_seed
from dataloader_bert import  DataSetLoaderBERT
from utils import compute_metrics
import json
import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np

set_seed(42)

model_name = "/M2/ArgosMHC_models/checkpoints/classic_t6_c3/checkpoint-102000/"  # mejor checkpoiunt
name_results = "predictions_esm2_t33_c5" # 
#pre_trained = "/M2/ArgosMHC_models/pre_trained_models/esm2_t33_650M_UR50D/"
pre_trained = "/M2/ArgosMHC_models/pre_trained_models/esm2_t6_8M_UR50D/"
dataset = "/M2/ArgosMHC_models/dataset/hlab/hlab_test_micro.csv"

model = BertRnn.from_pretrained(model_name, num_labels=2) # it fail for automodel for sequence classification
tokenizer = AutoTokenizer.from_pretrained(pre_trained)

seq_length = 50 # for MHC-I
test_dataset = DataSetLoaderBERT(dataset, tokenizer_name=pre_trained, max_length=seq_length)
data_iter = DataLoader(test_dataset, batch_size=16, shuffle=False)

print( tokenizer("YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGYLFGRDL", padding='max_length', max_length=seq_length) )
print(test_dataset[0])
print(type(test_dataset[0]))

# data_iter, es un dataLoader, de la base de datos de test, y tiene un batch_size de 16
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

model.eval() # is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn them off during model evaluation,

predictions = []
labels = []

with torch.no_grad(): # turn off gradients computation
    for i, batch in enumerate(data_iter): # por cada batch        
        labels.extend(batch['labels'].numpy())
        output = model(batch['input_ids'], batch['attention_mask']) # inference
        for row_sample in output.logits: # por cada muestra del batch
            logits = row_sample.numpy()
            probs = softmax(logits)
            predictions.append( [logits[0], logits[1], probs[0], probs[1]] )

#print(predictions)
df = pd.DataFrame(predictions, columns=["logit_class_0", "logit_class_1", "prob_class_0", "prob_class_1"])
df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
df['label'] = labels
print(df)

print(get_metrics(df['label'], df['prediction'], df['prob_class_1']))