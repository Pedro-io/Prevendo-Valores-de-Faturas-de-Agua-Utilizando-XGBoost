import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint
from scripts.pre_processing import *
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta


# Função para treinar o modelo para uma matrícula específica
def train_model_for_matricula(data, matricula):
    # Filtrar os dados para a matrícula selecionada
    matricula_data = data[data['MATRICULA'] == matricula].copy()
    matricula_data = matricula_data.drop('MATRICULA', axis=1)

    # Separar features (X) e target (y)
    X = matricula_data[['ANO_VENCIMENTO', 'MES_VENCIMENTO','TRIMESTRE', 'VALOR_FATURA_lag1', 'VALOR_FATURA_lag2']]
    y = matricula_data['VALOR_FATURA']

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pré-processamento 
    preprocessor = preprocessar_dados()
    X_train = preprocessor.fit_transform(X_train)  # Ajustar e transformar os dados de treinamento
    X_test = preprocessor.transform(X_test)  # Transformar os dados de teste

    # Modelo
    model = xgb.XGBRegressor(random_state=42)

    # Busca de hiperparâmetros (opcional, mas recomendado)
    param_distributions = {
        'colsample_bytree': [0.3, 0.5, 0.7],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }

    random_search = RandomizedSearchCV(
        model,  # Passar o modelo diretamente, não o Pipeline
        param_distributions,
        n_iter=3,  # Reduzi para ser mais rápido
        cv=3,  # Reduzi para ser mais rápido
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
    )

    # Treinar o modelo
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # Calcular o MAE no conjunto de teste
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return best_model, preprocessor, mae # Retorna o modelo, o preprocessor e o MAE


def prever_proximo_valor(model, preprocessor, data, matricula):
    # Filtrar os dados para a matrícula selecionada
    matricula_data = data[data['MATRICULA'] == matricula].copy()
    matricula_data = matricula_data.drop('MATRICULA', axis=1)

    # Ordenar por data
    matricula_data['DATA_VENCIMENTO'] = pd.to_datetime(matricula_data['DATA_VENCIMENTO'])
    matricula_data = matricula_data.sort_values(by='DATA_VENCIMENTO')

    # Data da última fatura
    ultima_data = matricula_data['DATA_VENCIMENTO'].iloc[-1]
    proxima_data = ultima_data + relativedelta(days=30)  # Usando relativedelta

    # Criar features para a próxima data
    ano_vencimento = proxima_data.year
    mes_vencimento = proxima_data.month
    trimestre = (mes_vencimento - 1) // 3 + 1

    # Criar DataFrame com as features
    proxima_data_df = pd.DataFrame({
        'ANO_VENCIMENTO': [ano_vencimento],
        'MES_VENCIMENTO': [mes_vencimento],
        'TRIMESTRE': [trimestre],
        'VALOR_FATURA_lag1': [matricula_data['VALOR_FATURA'].iloc[-1]],
        'VALOR_FATURA_lag2': [matricula_data['VALOR_FATURA'].iloc[-2] if len(matricula_data) > 1 else matricula_data['VALOR_FATURA'].mean()]
    })

    # Aplicar o mesmo pré-processador usado no treino
    X = preprocessor.transform(proxima_data_df)

    # Fazer a previsão
    prediction = model.predict(X)[0]
    return prediction, proxima_data


# Interface Streamlit
st.set_page_config(
    page_title="Previsão de Fatura de Água",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define as cores do tema
azul = '#0090FF'
branco = '#FFFFFF'

# Define o CSS personalizado
st.markdown(
    f"""
    <style>
    body {{
        color: black;
        background-color: {branco};
    }}
    .stApp {{
        background-color: {azul};
    }}
    .stButton > button {{
        color: {azul};
        border-color: {azul};
    }}
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stRadio > label,
    .stCheckbox > label {{
        color: {azul};
    }}
    </style>
    """,
    unsafe_allow_html=True
    
)

# Centralizar o título com CSS
st.markdown(
    """
    <style>
        .title {
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        .header-img {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 120px; /* Ajuste o tamanho da imagem conforme necessário */
            height: auto;
        }
    </style>
    <img src="https://prefeitura.pbh.gov.br/sites/default/files/logo-menu.svg" class="header-img">
    
    
    """,
    unsafe_allow_html=True
)

st.markdown("<br><br><br>", unsafe_allow_html=True)

st.markdown('<h1 class="title">Previsão de Fatura de Água</h1>', unsafe_allow_html=True)

# Explicação do modelo e das métricas
st.markdown("""
Este aplicativo usa um modelo de Machine Learning para prever o valor da próxima fatura de água de uma unidade da Prefeitura de Belo Horizonte a partir de uma matrícula selecionada. 
O modelo utiliza dados históricos de faturas, como o ano e mês de vencimento, o trimestre e os valores das faturas anteriores.

**Métricas de avaliação:**

*   **Mean Absolute Error (MAE):** O MAE mede a diferença média absoluta entre os valores previstos e os valores reais. 
    Quanto menor o MAE, melhor o modelo está performando.
""")

# Carregar os dados (uma vez)
@st.cache_data
def load_data():
    data = pd.read_csv('..\\data\\processed\\final_data.csv')
    data = remover_colunas(data)
    return data


data = load_data()

# Seletor de Matrícula
matriculas_disponiveis = data['MATRICULA'].unique().tolist()
matricula_selecionada = st.selectbox('Selecione a Matrícula:', matriculas_disponiveis)

# Botão para Iniciar a Previsão
if st.button('Prever Próxima Fatura'):
    with st.spinner(f'Treinando modelo para a matrícula {matricula_selecionada}...'):
        # Treinar o modelo para a matrícula selecionada
        model, preprocessor, mae = train_model_for_matricula(data, matricula_selecionada)

        if model is not None:
            # Fazer a previsão
            previsao, proxima_data = prever_proximo_valor(model, preprocessor, data, matricula_selecionada)

            # Exibir o Resultado
            st.write(f'A previsão para a próxima fatura da matrícula {matricula_selecionada} é: R$ {previsao:.2f}')

            # Exibir o MAE no conjunto de teste
            st.write(f"Mean Absolute Error (MAE) no conjunto de teste: {mae:.2f}")
            
            # Preparar os dados para o gráfico
            matricula_data_grafico = data[data['MATRICULA'] == matricula_selecionada].copy()
            matricula_data_grafico['DATA_VENCIMENTO'] = pd.to_datetime(matricula_data_grafico['DATA_VENCIMENTO'])
            matricula_data_grafico = matricula_data_grafico.sort_values(by='DATA_VENCIMENTO')

            # Adicionar a previsão ao DataFrame
            nova_linha = pd.DataFrame({'DATA_VENCIMENTO': [proxima_data], 'VALOR_FATURA': [previsao]})
            matricula_data_grafico = pd.concat([matricula_data_grafico, nova_linha], ignore_index=True)

            # Criar o gráfico usando Matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(matricula_data_grafico['DATA_VENCIMENTO'], matricula_data_grafico['VALOR_FATURA'], marker='o')
            ax.axvline(x=proxima_data, color='green', linestyle='--', linewidth=2, label=f"Previsão: R$ {previsao:.2f}")
            ax.set_title(f'Série Temporal da Matrícula {matricula_selecionada} com Previsão')
            ax.set_xlabel('Data de Vencimento')
            ax.set_ylabel('Valor da Fatura')
            ax.legend()
            st.pyplot(fig)

            # Exibir a tabela com os valores da matrícula
            st.subheader(f"Valores da Fatura da Matrícula {matricula_selecionada}")
            st.dataframe(matricula_data_grafico[['DATA_VENCIMENTO', 'VALOR_FATURA']], use_container_width=True)