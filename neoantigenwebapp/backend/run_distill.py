from model_utils_bert import  BertRnnDist
from transformers import BertConfig
from transformers import  AutoTokenizer
import torch

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

# ---- TEST ----
# el tokenizer devuelve los input_ids y el attention_mask como dos listas
#peptide,Label,Length,mhc
#LFGRDL,1,8,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
#RSDTPLIY,1,8,YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGY
#mhc+peptide CONCATENADOS
sample = tokenizer("YFAMYQENMAHTDANTLYIIYRDYTWVARVYRGYRSDTPLIY", padding='max_length', max_length=seq_length)

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

print("\n###########################################")
print("logits", output.logits)
print("result", (0 if output.logits[0] > output.logits[1] else 1))