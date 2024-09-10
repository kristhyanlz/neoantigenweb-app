from flask import Flask, request, jsonify
from flask_cors import CORS
import csv

from model_utils_bert import  BertRnnDist
from transformers import BertConfig
from transformers import  AutoTokenizer
import numpy as np
import torch

import pandas as pd

app = Flask(__name__)
CORS(app)

#DETECCION GPU
#Limitamos el uso de la VRAM
gpu_fraction = 1.0
# Verificamos si al menos tenemos 1 GPU compatible con CUDA
if torch.cuda.is_available():
  # Apuntamos a la primera GPU CUDA
  torch.cuda.set_per_process_memory_fraction(gpu_fraction)
  device = torch.device("cuda")
  print('# GPUs: ', torch.cuda.device_count())
  print('Nombre de la GPU:', torch.cuda.get_device_name(0))
# Si no tenemos alguna GPU CUDA
else:
  print('Se utilizará la CPU')
  device = torch.device("CPU")

model_name = "../../esm2_distilbert_t33_c3/checkpoint-201000/"  # mejor checkpoiunt
name_results = "new_predict" # nombre de los archivos donde se guardara los resultados. 
pre_trained = "../../esm2_t33_650M_UR50D" # Modelo original

config = BertConfig.from_pretrained(model_name, num_labels=2 )
model = BertRnnDist.from_pretrained(model_name, config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained(pre_trained)

seq_length = 50 # for MHC-I

def softmax(x):
  """Compute softmax values for each sets of scores in x."""
  result = np.exp(x) / np.sum(np.exp(x), axis=0)
  return result.tolist()[1]

@app.route('/predict', methods=['POST'])
def predict():
  global tokenizer, seq_length, model
  data = request.get_json()
  
  if type(data) == dict:
    data = [data]

  result = []
  for item in data:
    testStr = item['mhc'] + item['peptide']
    sample = tokenizer(testStr, padding='max_length', max_length=seq_length)
    # convertimos en tensor, debe ser lista de listas
    if torch.cuda.is_available():
      ids = torch.cuda.IntTensor([sample['input_ids']]) # tensor 2D
      masks = torch.cuda.IntTensor([sample['attention_mask']]) # tensor 2D
    else:
      ids = torch.IntTensor([sample['input_ids']]) # tensor 2D
      masks = torch.IntTensor([sample['attention_mask']]) # tensor 2D

    model.eval()
    with torch.no_grad(): # turn off gradients computation
      output = model(ids, masks )

    result.append({
      'hla': item['hla'],
      'peptide': item['peptide'],
      'mhc': item['mhc'],
      'prediction': (0 if output.logits[0] > output.logits[1] else 1),
      'score': softmax(output.logits.cpu().numpy())
    }) 
  print(result)
  return jsonify(result)

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')