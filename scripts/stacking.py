#%%
import pandas as pd

columns= [
        "MATRICULA",
        "VALOR_FATURA",
        "VOLUME_FATURA_AGUA",
        "VOLUME_FATURA_ESGOTO",
        "VOLUME_MEDIDO_AGUA",
        "VOLUME_MEDIDO_ESGOTO",
        "NOME_CLIENTE",
        "HIDROMETRO",
        "DATA_VENCIMENTO",
        "TIPO_LOGRADOURO",
        "NOME_LOGRADOURO",
        "NUM_IMOVEL",
        "COMPLEMENTO",
        "INFO_COMPLEMENTO",
        "BAIRRO",
        "residuo"
]

files= []
for year in range(2022, 2025):
    for month in range(1, 13):
        if year == 2024 and month == 12:
            break
        files.append(f"{year}{month:02d}_PBH.txt")

file_list= []
for file in files:
        df= pd.read_csv(f"../../data/raw/{file}", sep=";", encoding="latin1", header=None)
        #print(f"Arquivo: {file} - Vencimento: {df[8].unique()}") # Check em DATA_VENCIMENTO
        file_list.append(df)

df= pd.concat(file_list, ignore_index=True)
df.columns= columns
df = df.drop(columns=["residuo"])

#df.to_csv("../../data/processed/stacked_water.csv", index=False, encoding="utf-8")
