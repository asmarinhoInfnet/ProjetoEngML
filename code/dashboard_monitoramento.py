import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, classification_report, confusion_matrix

prod_base = pd.read_parquet('../data/processed/prediction_prod.parquet')
dev_base = pd.read_parquet('../data/processed/prediction_final_dev.parquet')

st.title(f"""
Sistema de monitoramento de previsão de acertos - Kobe Bryant
""")

st.markdown(f"""
Esta interface pode ser utilizada para a avaliação e análise de dados para o modelos de previsão de acertos escolhido.
""")

st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Monitoramento da saúde do modelo ao longo do seu ciclo de vida e entrada de dados para avaliação segundo modelo criado.
""")

st.sidebar.header('Resultados de Log Loss e F1 Score para a base de treinamento submetida ao modelo FINALIZADO.')

data = [1 if x > 0.5 else 0 for x in dev_base.prediction_score_1]
st.sidebar.write('Log Loss:')
st.sidebar.write(log_loss(dev_base['shot_made_flag'], data))
st.sidebar.write('F1 Score:')
st.sidebar.write(f1_score(dev_base['shot_made_flag'], data))

st.sidebar.header('Resultados de Log Loss e F1 Score para a base de produção submetida ao modelo.')

st.sidebar.write('Log Loss:')
st.sidebar.write(log_loss(prod_base['shot_made_flag'], prod_base.predict_score))
st.sidebar.write('F1 Score:')
st.sidebar.write(f1_score(prod_base['shot_made_flag'], prod_base.predict_score))

# st.write(dev_base)
# st.write(prod_base)


st.markdown(f"""
O primeiro gráfico mostra a comparação para o conjunto de treinamento, após o modelo finalizado.
""")

fignum = plt.figure(figsize=(6,4))
sns.distplot(dev_base.prediction_score_1,
    label='Dados históricos',
    ax = plt.gca())

sns.distplot(dev_base['shot_made_flag'],
    label='Dados atuais',
    ax = plt.gca())

plt.title('Distribuição de arremessos estimados e reais de Kobe Bryant para os dados de TREINAMENTO')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade Acerto')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)

st.markdown(f"""
O gráfico a seguir mostra o uso do modelo testado para a estimação a partir dos dados de produção.
""")

fignum = plt.figure(figsize=(6,4))
sns.distplot(dev_base.prediction_score_1,
    label='Dados históricos',
    ax = plt.gca())

sns.distplot(prod_base.predict_score,
    label='Dados atuais',
    ax = plt.gca())

plt.title('Distribuição da ESTIMAÇÃO para arremessos de Kobe Bryant')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade Acerto')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)

st.markdown(f"""
Por fim, é exibida a comparação entre os dados reais do conjunto de arremessos de produção e o resultado predito.
""")

fignum = plt.figure(figsize=(6,4))
sns.distplot(prod_base.predict_score,
    label=' Estimação',
    ax = plt.gca())

sns.distplot(prod_base['shot_made_flag'],
    label='Realidade',
    ax = plt.gca())

plt.title('Distribuição de arremessos estimados e reais de Kobe Bryant para os dados de PRODUÇÃO')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade Acerto')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')
st.pyplot(fignum)