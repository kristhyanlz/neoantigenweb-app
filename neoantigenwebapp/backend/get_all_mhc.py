import pandas as pd
df = pd.read_csv('../../hlab_train.csv', header=None, low_memory=False)
#unique_values = list(df['HLA'].values)
#unique_values = set(unique_values[1:])

# 1: Todas las filas menos el header, 0 Primera columna
unique_values = set(df.iloc[1:, 0])
unique_list = sorted(list(unique_values))
print(f"UNIQUE_LIST: \n {unique_list} \n {type(unique_list)}")


#unique_df = pd.DataFrame(unique_values, columns=['HLA'])
#unique_df.to_csv('unique_hla.csv', index=False, header=False)
#print(unique_df)


with open('unique_hla.js', 'w') as js_file:
  js_file.write('export const hlaData = [\n')
  for value in unique_list:
    js_file.write(f'  "{value}",\n')
  js_file.write('];\n')