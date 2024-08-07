from flask import Flask, request, jsonify
import csv

from model_utils_bert import BertRnn, BertRnnDist
from transformers import Trainer, TrainingArguments, BertConfig
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from dataloader_bert import DataSetLoaderBERT
from utils import compute_metrics
import json

import pandas as pd

app = Flask(__name__)

model_name = "../../esm2_distilbert_t33_c3/checkpoint-201000/"  # mejor checkpoiunt
name_results = "new_predict" # nombre de los archivos donde se guardara los resultados.

seq_length = 50 # for MHC-I
config = BertConfig.from_pretrained(model_name, num_labels=2 )

model = Trainer(model = BertRnnDist.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)

#HLA,peptide,Label,Length,mhc
#HLA-A*01:01,LFGRDL,1,8,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY

@app.route('/predict', methods=['POST'])
def predict():
  global model_name, name_results, seq_length, config, model
  data = request.get_json()
  print(data)
  print(config)
  with open('data.csv', mode='w') as file:
    writer = csv.DictWriter(file, fieldnames=["HLA", "peptide", "Label", "Length", "mhc"])
    writer.writeheader()
    writer.writerows(data)

  test_dataset = DataSetLoaderBERT("./data.csv", tokenizer_name="../../esm2_t33_650M_UR50D", max_length=seq_length)
  predictions, label_ids, metrics = model.predict(test_dataset)

  print(metrics)
  #f = open(name_results + ".txt", "w")
  #f.write(model_name + "\n")
  #f.write(json.dumps(metrics))
  #f.close()

  ########################## print predictions #######################################
  df = pd.DataFrame(predictions)

  df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
  df['label'] = label_ids
  print(df)
  df.to_csv(name_results + ".csv")
  return df.to_json()

"""
model_name = "../../esm2_distilbert_t33_c3/checkpoint-201000/"  # mejor checkpoiunt
name_results = "new_predict" # nombre de los archivos donde se guardara los resultados. 

seq_length = 50 # for MHC-I
config = BertConfig.from_pretrained(model_name, num_labels=2 )
print(config)

model = Trainer(model = BertRnnDist.from_pretrained(model_name, config=config), compute_metrics = compute_metrics)
test_dataset = DataSetLoaderBERT("../../hlab_train_micro.csv", tokenizer_name="../../esm2_t33_650M_UR50D", max_length=seq_length)
predictions, label_ids, metrics = model.predict(test_dataset)
print(model_name)
print(metrics)
f = open(name_results + ".txt", "w")
f.write(model_name + "\n")
f.write(json.dumps(metrics))
f.close()

########################## print predictions #######################################
import pandas as pd
df = pd.DataFrame(predictions)

df['prediction'] = df.apply(lambda row: ( 0 if row[0] > row[1] else 1 ), axis=1)
df['label'] = label_ids
print(df)
df.to_csv(name_results + ".csv")
"""

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')