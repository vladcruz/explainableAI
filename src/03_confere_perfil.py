'''
Arquivo.....: 03_explica_lime.py
Data........: 23/09/2026
Autor.......: Vladmir B. da Cruz
Contato.....: vladcruz@gmail.com
Descrição...: Explicabilidade Local (XAI) com LIME sobre a base clusterizada.
              Estruturado para adaptar automaticamente a qualquer número K de perfis.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
import warnings

# Silenciando warnings para manter a tela limpa
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração visual do Seaborn
sns.set_theme(style="whitegrid")

# ==========================================
# PARTE A: CARREGAMENTO DOS DADOS CLUSTERIZADOS
# ==========================================
ARQUIVO_CLUSTERIZADO = "base_acme_co_clusterizada.csv"
print(f"Carregando a base agrupada: '{ARQUIVO_CLUSTERIZADO}'...\n")

try:
    df_acme = pd.read_csv(ARQUIVO_CLUSTERIZADO)
except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_CLUSTERIZADO}' não foi encontrado. Execute o Script 2 primeiro.")
    exit()

# Features completas para o classificador e o LIME avaliarem
features_completas = [
    'Renda_Comprovada', 
    'Tempo_Cliente_Meses', 
    'Tempo_Pagamento_Dias', 
    'Comprometimento_Renda_Pct', 
    'Valor_Solicitado'
]

X = df_acme[features_completas]
y = df_acme['Cluster']

# ==========================================
# PARTE B: DETECÇÃO DINÂMICA DE K E NOMENCLATURA DAS CLASSES
# ==========================================
# Detecta a quantidade real de clusters gerada no Script 2
k_total = df_acme['Cluster'].nunique()

# Mapeia dinamicamente os nomes dos perfis em ordem crescente de cluster (0 a K-1)
mapeamento_perfis = df_acme[['Cluster', 'Perfil_Cliente']].drop_duplicates().sort_values('Cluster')
nomes_classes = mapeamento_perfis['Perfil_Cliente'].tolist()

print(f"Clusters identificados na base: K = {k_total}")
for cid, cnome in zip(mapeamento_perfis['Cluster'], nomes_classes):
    print(f"  -> Cluster {cid}: {cnome}")
print()

# ==========================================
# PARTE C: TREINAMENTO DO MODELO CLASSIFICADOR
# ==========================================
print("Treinando modelo de classificação (Random Forest) para validação das fronteiras...\n")
modelo_classificacao = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_classificacao.fit(X, y)

# ==========================================
# PARTE D: REPRESENTANTES MÉDIOS E CONFIGURAÇÃO DO LIME
# ==========================================
# Centroide empírico de cada grupo para criar os 'Clientes Representantes'
clientes_medios = df_acme.groupby('Cluster')[features_completas].mean().sort_index()

# Instanciando o explicador LIME configurado dinamicamente com as classes encontradas
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X.values,
    feature_names=features_completas,
    class_names=nomes_classes,
    mode='classification',
    random_state=42
)

# ==========================================
# PARTE E: GERANDO O DASHBOARD COMPARATIVO LIME (GRADE DINÂMICA)
# ==========================================
print("=== GERANDO O DASHBOARD DE EXPLICABILIDADE LOCAL (LIME) ===")
print(f"Construindo layout dinâmico para {k_total} perfis de cliente...")

# Cálculo dinâmico de linhas e colunas para a grade de visualização
n_cols = 2 if k_total <= 4 else 3
n_rows = int(np.ceil(k_total / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.5 * n_cols, 5.5 * n_rows))
fig.suptitle("Dashboard Explainable AI (LIME): Regras de Decisão por Perfil de Cliente", fontsize=16, y=1.01)

# Garante que 'axes' seja sempre um array 1D iterável
axes_flat = np.atleast_1d(axes).flatten()

for cluster_id in range(k_total):
    dados_cliente_medio = clientes_medios.loc[cluster_id].values
    
    # Gerando a explicação local para o cliente representativo
    exp = explainer_lime.explain_instance(
        data_row=dados_cliente_medio, 
        predict_fn=modelo_classificacao.predict_proba, 
        num_features=len(features_completas), 
        top_labels=1
    )
    
    # Classe prevista pelo modelo substituto
    classe_prevista = exp.available_labels()[0]
    nome_classe_prevista = nomes_classes[classe_prevista]
    
    # Extração das regras e seus respectivos pesos
    explicacao_lista = exp.as_list(label=classe_prevista)
    regras = [item[0] for item in explicacao_lista]
    pesos = [item[1] for item in explicacao_lista]
    
    # Invertendo a ordem para que a feature com maior peso fique no topo
    regras.reverse()
    pesos.reverse()
    
    # Semântica visual: Verde = apoia a classificação | Vermelho = contradiz
    cores = ['#2ca02c' if peso > 0 else '#d62728' for peso in pesos]
    
    ax = axes_flat[cluster_id]
    barras = ax.barh(regras, pesos, color=cores, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
    
    ax.set_title(f"Cluster {cluster_id}: {nome_classe_prevista}", fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel("Impacto na Probabilidade (LIME)", fontsize=10)
    
    # Rótulos de dados numéricos ao lado das barras
    for barra in barras:
        largura = barra.get_width()
        pos_x = largura + 0.01 if largura >= 0 else largura - 0.01
        alinhamento = 'left' if largura >= 0 else 'right'
        ax.text(pos_x, barra.get_y() + barra.get_height() / 2, f'{largura:+.2f}', 
                va='center', ha=alinhamento, fontsize=9.5, fontweight='bold')

# Remove subplots excedentes (se a grade tiver mais células que clusters)
for j in range(k_total, len(axes_flat)):
    fig.delaxes(axes_flat[j])

plt.tight_layout(w_pad=3.0, h_pad=3.0)
plt.show()

print("\nDashboard LIME finalizado com sucesso!")