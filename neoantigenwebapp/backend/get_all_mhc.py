import pandas as pd
import json

# Cargar el CSV
df = pd.read_csv('../../hlab_train.csv', low_memory=False)

# Eliminar duplicados y ordenar por HLA
unique_values = df[['HLA', 'mhc']].drop_duplicates().sort_values(by='HLA')

# Crear la lista con el formato específico
unique_list = [{'label': row['HLA'], 'id': row['mhc']} for _, row in unique_values.iterrows()]

# Convertir la lista a JSON
unique_json = json.dumps(unique_list, indent=2)

print("\nHLA-mhc únicos:", len(unique_list))

# Guardar la salida en un archivo JS
with open('unique_hla.js', 'w') as js_file:
    js_file.write(f'export const hlaData = {unique_json}')



"""
import pandas as pd
import json
df = pd.read_csv('../../hlab_train.csv', low_memory=False)

print(df.columns)

# 1: Todas las filas menos el header, 0 Primera columna
unique_values = df[['HLA', 'mhc']].drop_duplicates().sort_values(by='HLA')
print(f"UNIQUE_VALUES: \n {unique_values} \n {type(unique_values)}")

unique_list = unique_values.to_dict(orient='records')
#print(f"UNIQUE_LIST: \n {unique_list} \n {type(unique_list)}")

unique_json = json.dumps(unique_list, indent=2)

#print(unique_json)
print("\nHLA-mhc unicos:",len(unique_list))

with open('unique_hla.js', 'w') as js_file:
  js_file.write(f'export const hlaData = {unique_json}')
"""