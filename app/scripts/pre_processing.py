# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 



def separar_base(data): 
    """Separa a base entre treino validação, treino e teste 

    Args:
        data (dataframe): dataframe pandas com os dados 

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = data.drop(columns=["VALOR_FATURA"])
    y = data["VALOR_FATURA"]

    # treino+validação e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # treino+validação em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    # A validação será 20% (0.25 * 0.8) dos dados totais.

    print("Tamanho do treino:", X_train.shape)
    print("Tamanho da validação:", X_val.shape)
    print("Tamanho do teste:", X_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
def remover_colunas(data): 
    """Remove colunas que não são relevantes para nosso modelo 
    """
    drop_features = [
                    'Unnamed: 0', # irrelevante
                     'VOLUME_FATURA_AGUA','VOLUME_FATURA_ESGOTO', 'VOLUME_MEDIDO_AGUA', 'VOLUME_MEDIDO_ESGOTO',#Removendo colunas para evitar o data leakage
                     'BAIRRO'# Estamos removendo o bairro por sua alta dimensionalide 
                     ]
    data = data.drop(columns = drop_features)
    # REMOVENDO FATURAS ZERADAS
    data = data[data['VALOR_FATURA'] != 0]
    
    return data 


def preprocessar_dados():

    """
    Transformer para pré-processamento dos dados. Recebe um dataframe como parametro e aplica as transformações
    listadas. 
    
    """    
        
    # Selecionando variaveis numericas, categoricas e variaveis que devem ser removidas para evitar data leakage
    numerical_features = ['VALOR_FATURA_lag1','VALOR_FATURA_lag2']
    ordinal_features = ['ANO_VENCIMENTO', 'MES_VENCIMENTO']

    
   
    preprocessor = ColumnTransformer(
        transformers=[
            # Aplica escalonamento aos dados numéricos.
            ('num', StandardScaler(), numerical_features),
        
        ])

    
    return preprocessor


