'''
Arquivo.....: 01_gera_base.py
Data........: 20/09/2026
Autor.......: Vladmir B. da Cruz
Contato.....: vladcruz@gmail.com
Descrição...: Script que gera a base de dados e já faz a análise SHAP. 
'''

import os
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Ignorando o FutureWarning gerado por bibliotecas como o SHAP ao usar versões mais recentes do NumPy
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração visual do Seaborn para os gráficos ficarem atraentes
sns.set_theme(style="whitegrid", palette="muted")

# Nome do arquivo da nossa base de dados
ARQUIVO_BASE = "base_acme_co.csv"

# ==========================================
# PARTE A: CARREGAMENTO OU GERAÇÃO DA BASE DE DADOS
# ==========================================

if os.path.exists(ARQUIVO_BASE):
    # 1. Se o arquivo já existir, apenas fazemos a leitura
    print(f"Arquivo '{ARQUIVO_BASE}' encontrado. Carregando base existente...")
    df_acme = pd.read_csv(ARQUIVO_BASE)
else:
    # 2. Se o arquivo não existir, geramos os dados
    print(f"Arquivo '{ARQUIVO_BASE}' não encontrado. Gerando nova base sintética...")
    
    n_clientes = 10000
    
    # Utilizando a API do NumPy (Generator) configurando a seed para repetibilidade
    # Note que a Seed é especial! :D
    rng = np.random.default_rng(seed=42)

    # Gerando dados hipotéticos
    ids = np.arange(1, n_clientes + 1)
    nomes = [f"Cliente_{i}" for i in ids]
    estados = rng.choice(['SP', 'RJ', 'MG', 'PR', 'SC'], n_clientes)

    # Renda comprovada (R$ 2.000 a R$ 25.000)
    rendas = rng.normal(7000, 3000, n_clientes)
    rendas = np.clip(rendas, 2000, 25000)

    # Tempo como cliente (em meses, de 1 a 120)
    tempo_cliente = rng.integers(1, 120, endpoint=True, size=n_clientes)

    # Tempo médio de pagamento nos últimos 6 meses (em dias)
    # Assumindo que o vencimento é dia 10. Negativos: pagamento antecipado. Positivos: atraso.
    tempo_pagamento = rng.normal(2, 8, n_clientes) 

    # Comprometimento da renda (em %, de 10% a 70%)
    comprometimento = rng.normal(35, 15, n_clientes)
    comprometimento = np.clip(comprometimento, 10, 70)

    # Valor do Crédito Solicitado (R$ 1.000 a R$ 100.000)
    valor_solicitado = rng.normal(15000, 10000, n_clientes)
    valor_solicitado = np.clip(valor_solicitado, 1000, 100000)

    # Montando o DataFrame
    df_acme = pd.DataFrame({
        'ID': ids,
        'Nome': nomes,
        'Estado': estados,
        'Renda_Comprovada': rendas,
        'Tempo_Cliente_Meses': tempo_cliente,
        'Tempo_Pagamento_Dias': tempo_pagamento,
        'Comprometimento_Renda_Pct': comprometimento,
        'Valor_Solicitado': valor_solicitado
    })

    # Criando o 'Risco_Simulado' (Variável Alvo para o modelo entender o que é ruim/bom)
    # Essa é uma definição nossa, pode ser configurada de acordo com a regra de negócio
    df_acme['Risco_Simulado'] = (
        (df_acme['Comprometimento_Renda_Pct'] * 0.4) + 
        (df_acme['Tempo_Pagamento_Dias'] * 2.5) + 
        ((df_acme['Valor_Solicitado'] / df_acme['Renda_Comprovada']) * 5)
    )
    # Normalizando de 0 a 100
    df_acme['Risco_Simulado'] = 100 * (df_acme['Risco_Simulado'] - df_acme['Risco_Simulado'].min()) / (df_acme['Risco_Simulado'].max() - df_acme['Risco_Simulado'].min())

    # Exportando a base para registro e uso com outras ferramentas
    df_acme.to_csv(ARQUIVO_BASE, index=False)
    print("Nova base gerada e exportada com sucesso!")

# ==========================================
# PARTE B: PREPARAÇÃO PARA O SHAP
# ==========================================
# Separando variáveis preditivas (X) do nosso alvo (y)
X = df_acme.drop(columns=['ID', 'Nome', 'Risco_Simulado'])
le = LabelEncoder()
X['Estado'] = le.fit_transform(X['Estado']) # Transformando Texto em Número
y = df_acme['Risco_Simulado']

# ==========================================
# PARTE C: TREINANDO O MODELO E EXPLICANDO COM SHAP
# ==========================================
print("Treinando o modelo RandomForest e calculando valores SHAP...")
modelo_base = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_base.fit(X, y)

# Instanciando o explicador SHAP
explainer = shap.TreeExplainer(modelo_base)
shap_values = explainer.shap_values(X)

# Criando o gráfico de resumo do SHAP
plt.figure(figsize=(10, 6))
plt.title("Explainable AI (SHAP) - Variáveis que mais impactam o Risco na ACME Co.", fontsize=14, pad=20)
shap.summary_plot(shap_values, X, plot_type="dot", show=False)
plt.tight_layout()
plt.show()

print("Script 1 Finalizado. Base criada e valores SHAP definidos!")