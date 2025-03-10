#%% 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import  OneHotEncoder


pd.set_option('display.max_rows', None)

#%% 
data = pd.read_csv("..\\..\\data\\processed\\processed_water.csv")
# %%
data.info()


#%% 
# Deixando no typo correto 
data[['VALOR_FATURA','VOLUME_FATURA_AGUA','VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO']] = data[['VALOR_FATURA','VOLUME_FATURA_AGUA','VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO']].astype('float64')
data[['BAIRRO', 'MATRICULA']] = data[['BAIRRO', 'MATRICULA']].astype('category')
data['DATA_VENCIMENTO'] = pd.to_datetime(data['DATA_VENCIMENTO'])
# %%
# Verificando os valores em todas as colunas
colunas = data.columns
for col in colunas: 
    print(f"Valores da coluna {col}")
    print()
    print(data[col].value_counts())
    print("-"*40)

# %%
# Removendo todas as faturas que estão zeradas:
data = data[data['VALOR_FATURA'] != 0]
data.head()
data.info()
# %%

# Extraindo Ano e Mês de vencimento das faturas

data['ANO_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.year
data['MES_VENCIMENTO'] = data['DATA_VENCIMENTO'].dt.month
data['TRIMESTRE'] = data['DATA_VENCIMENTO'].dt.quarter

#%%[markdown]
# Vamos criar lags para faciliatar o nosso modelo a identificar padrões das faturas. 
# Como queremos prever o valor futuro da próxima fatura, é interessante que o modelo saiba qual foi o valor passado desta fatura,
# e para isso utilizaremos valores passados das faturas de cada matícula.  
#%%
def criar_lags_por_matricula(data, col, id_col, max_lag=5):
    """
    Cria colunas de lag para séries temporais agrupadas por um identificador.
    
    Parameters:
    - data (pd.DataFrame): DataFrame contendo os dados.
    - col (str): Nome da coluna para a qual os lags serão criados.
    - id_col (str): Nome da coluna de identificação para agrupar os dados (ex.: matrícula).
    - max_lag (int): Número máximo de lags a serem criados.
    
    Returns:
    - pd.DataFrame: DataFrame com as colunas de lag adicionadas.
    """
    def criar_lags_grupo(grupo):
        for lag in range(1, max_lag + 1):
            grupo[f'{col}_lag{lag}'] = grupo[col].shift(lag)
        return grupo
    
    # Aplica a função de lags para cada grupo
    data = data.groupby(id_col).apply(criar_lags_grupo)
    
    # Remove valores nulos gerados pelos lags
    data = data.dropna().reset_index(drop=True)
    
    return data


# Criar lags
data = criar_lags_por_matricula(data, col='VALOR_FATURA', id_col='MATRICULA', max_lag=2)

# %% Ordenando por data
data = data.sort_values(by=['DATA_VENCIMENTO'], ascending=[True])

#%% [markdown]
# > Como Veremos abaixo, existem diversos valores onde todos os valores correspondentes 
# > ao volume de **AGUA** e **ESGOTO** estão zerados, porém temos valores para as faturas nas mesmas instâncias
# > oque pode caracterizar inconsistencias nos dados. 
# > Se possível, é recomendavel validar este tipo de inconsistencia com um especialista do domínio em questão.  
# %%
 
data_zero = data[
    (data['VOLUME_FATURA_AGUA'] == 0) &
    (data['VOLUME_MEDIDO_AGUA'] == 0) &
    (data['VOLUME_FATURA_ESGOTO'] == 0) &
    (data['VOLUME_MEDIDO_ESGOTO'] == 0)]

data_zero.head()
# %%
data_zero['VALOR_FATURA'].value_counts()
# %%
data.to_csv("..\\..\\data\\processed\\final_data.csv")

# %%
