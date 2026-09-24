'''
Arquivo.....: 02_agrupa_base.py
Data........: 23/09/2026
Autor.......: Vladmir B. da Cruz
Contato.....: vladcruz@gmail.com
Descrição...: Script que utiliza a base gerada pelo Script 1 para agrupamento via K-Means,
              destacando o K ideal de forma sincronizada na Inércia e na Silhueta.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

# Silenciando warnings para manter a saída limpa
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Configuração visual do Seaborn
sns.set_theme(style="whitegrid", palette="muted")

# ==========================================
# PARTE A: CARREGAMENTO DOS DADOS E PREPARAÇÃO
# ==========================================
ARQUIVO_BASE = "base_acme_co.csv"
print("Carregando os dados da ACME Co. para agrupamento...\n")

try:
    df_acme = pd.read_csv(ARQUIVO_BASE)
except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_BASE}' não foi encontrado. Rode o Script 1 primeiro.")
    exit()

# Features selecionadas baseadas na explicação global do SHAP (Script 1)
features_cluster = [
    'Tempo_Pagamento_Dias', 
    'Comprometimento_Renda_Pct', 
    'Valor_Solicitado', 
    'Renda_Comprovada'
]

X_cluster = df_acme[features_cluster]

# Padronização (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# ==========================================
# PARTE B: ANÁLISE DE HIPERPARÂMETROS (INÉRCIA E SILHUETA)
# ==========================================
print("=== AVALIAÇÃO DE HIPERPARÂMETROS: INÉRCIA E SILHUETA ===")
print("Objetivo: Identificar o K ideal ponderando compacidade interna e separação entre grupos.\n")

valores_k = list(range(1, 11))
inercia = []
silhuetas = []

for k in valores_k:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    inercia.append(kmeans_temp.inertia_)
    
    # Silhueta exige no mínimo 2 clusters
    if k > 1:
        silhuetas.append(silhouette_score(X_scaled, labels_temp))
    else:
        silhuetas.append(np.nan)

# Tabela consolidada de métricas
df_metricas = pd.DataFrame({
    'K (Grupos)': valores_k,
    'Inércia (Erro)': inercia,
    'Silhueta Média': silhuetas
})

inercia_k1 = inercia[0]
df_metricas['Queda_Marginal'] = df_metricas['Inércia (Erro)'].pct_change() * 100
df_metricas['Queda_Total'] = ((df_metricas['Inércia (Erro)'] - inercia_k1) / inercia_k1) * 100

# -------------------------------------------------------------
# DEFINIÇÃO DO K IDEAL (Variável centralizada)
# -------------------------------------------------------------
# Por padrão, assume o K que maximiza a silhueta média.
# Você pode sobrescrever manualmente (ex: k_ideal = 4) se preferir guiar a aula.
idx_melhor_silhueta = df_metricas['Silhueta Média'].idxmax()
k_ideal = int(df_metricas.loc[idx_melhor_silhueta, 'K (Grupos)'])

# Resgatando os valores de Inércia e Silhueta correspondentes ao k_ideal escolhido
idx_k = valores_k.index(k_ideal)
inercia_k_ideal = inercia[idx_k]
silhueta_k_ideal = silhuetas[idx_k]

# Formatação e exibição da tabela no terminal
df_display = df_metricas.copy()
df_display['Inércia (Erro)'] = df_display['Inércia (Erro)'].round(2)
df_display['Silhueta Média'] = df_display['Silhueta Média'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
df_display['Queda_Marginal'] = df_display['Queda_Marginal'].fillna(0).round(2).astype(str) + '%'
df_display['Queda_Total'] = df_display['Queda_Total'].round(2).astype(str) + '%'

print("Tabela Comparativa de Métricas:")
print(df_display.to_string(index=False))

print(f"\n--- Conclusão Matemática ---")
print(f"-> K Ideal Selecionado: K={k_ideal}")
print(f"-> Inércia no ponto: {inercia_k_ideal:,.2f}")
print(f"-> Silhueta no ponto: {silhueta_k_ideal:.4f} (Maior separação global entre grupos)\n")

# ==========================================
# PLOTAGEM COMPARATIVA COM K IDENTIFICADO EM AMBOS OS GRÁFICOS
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# 1. Gráfico do Cotovelo (Inércia)
ax1.plot(valores_k, inercia, marker='o', linestyle='--', color='#1f77b4', markersize=7, label='Inércia (WCSS)')
ax1.plot(k_ideal, inercia_k_ideal, marker='o', markersize=11, color='#d62728', label=f'K Ideal Selecionado (K={k_ideal})')
ax1.axvline(x=k_ideal, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8)

# Anotação com seta no cotovelo
ax1.annotate(
    f'Cotovelo (K={k_ideal})\nInércia: {inercia_k_ideal:,.1f}',
    xy=(k_ideal, inercia_k_ideal),
    xytext=(k_ideal + 0.8, inercia_k_ideal + (max(inercia) * 0.08)),
    arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
    fontsize=10,
    fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=1)
)

ax1.set_title("Método do Cotovelo (Compacidade)", fontsize=13, pad=10)
ax1.set_xlabel("Número de Clusters (K)", fontsize=11)
ax1.set_ylabel("Inércia (WCSS)", fontsize=11)
ax1.set_xticks(valores_k)
ax1.legend(loc='upper right')

# 2. Gráfico da Silhueta
ax2.plot(valores_k[1:], silhuetas[1:], marker='s', linestyle='-', color='#2ca02c', markersize=7, label='Silhueta Média')
if pd.notnull(silhueta_k_ideal):
    ax2.plot(k_ideal, silhueta_k_ideal, marker='s', markersize=11, color='#d62728', label=f'K Ideal Selecionado (K={k_ideal})')
    
    # Anotação com seta no pico da silhueta
    ax2.annotate(
        f'Pico da Silhueta (K={k_ideal})\nScore: {silhueta_k_ideal:.4f}',
        xy=(k_ideal, silhueta_k_ideal),
        xytext=(k_ideal + 0.8, silhueta_k_ideal - 0.05 if silhueta_k_ideal > 0.3 else silhueta_k_ideal + 0.05),
        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", lw=1)
    )

ax2.axvline(x=k_ideal, color='#d62728', linestyle=':', linewidth=1.8, alpha=0.8)
ax2.set_title("Coeficiente de Silhueta (Separação)", fontsize=13, pad=10)
ax2.set_xlabel("Número de Clusters (K)", fontsize=11)
ax2.set_ylabel("Score de Silhueta [-1 a +1]", fontsize=11)
ax2.set_xticks(valores_k[1:])
ax2.legend(loc='lower right')

plt.suptitle(f"Diagnóstico Integrado de Clusters: K={k_ideal} como Ponto de Equilíbrio", fontsize=15, y=1.03)
plt.tight_layout()
plt.show()

# ==========================================
# PARTE C: APLICANDO O K-MEANS COM O K DEFINIDO
# ==========================================
print(f"Executando o K-Means final configurado com K = {k_ideal} grupos...\n")

kmeans = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
df_acme['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# PARTE D: ORDENAÇÃO DOS GRUPOS E ROTULAGEM DINÂMICA
# ==========================================
# Ordenação pelo Risco Simulado médio (menor risco -> maior risco)
perfil_clusters = df_acme.groupby('Cluster')[features_cluster + ['Risco_Simulado']].mean()
perfil_clusters = perfil_clusters.sort_values(by='Risco_Simulado')

mapa_ordenacao = {antigo_id: novo_id for novo_id, antigo_id in enumerate(perfil_clusters.index)}
df_acme['Cluster'] = df_acme['Cluster'].map(mapa_ordenacao)

# Rotulagem adaptativa
mapa_tags = {i: f"Grupo {i+1} (Risco Nível {i+1})" for i in range(k_ideal)}

df_acme['Perfil_Cliente'] = df_acme['Cluster'].map(mapa_tags)
ordem_perfis = [mapa_tags[i] for i in range(k_ideal)]

# Paleta semântica proporcional (de verde a vermelho)
cores_array = sns.color_palette("RdYlGn_r", n_colors=k_ideal).as_hex()
cores_perfil = {mapa_tags[i]: cores_array[i] for i in range(k_ideal)}

print(f"=== RESUMO DAS MÉDIAS POR PERFIL IDENTIFICADO (K={k_ideal}) ===")
resumo = df_acme.groupby(['Cluster', 'Perfil_Cliente'])[features_cluster + ['Risco_Simulado']].mean().round(2)
print(resumo)
print("-" * 55)

# ==========================================
# PARTE E: VISUALIZAÇÃO MULTIDIMENSIONAL DOS CLUSTERS
# ==========================================
print("\nGerando visualizações multidimensionais...")

# 1. PairPlot
g = sns.pairplot(
    df_acme, 
    vars=features_cluster, 
    hue='Perfil_Cliente', 
    hue_order=ordem_perfis,
    palette=cores_perfil,
    plot_kws={'alpha': 0.6, 's': 30},
    diag_kind='kde'
)
g.figure.suptitle(f"Análise Multidimensional dos Perfis de Clientes (K={k_ideal})", y=1.02, fontsize=15)
plt.show()

# 2. Boxplots das Variáveis-Chave
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Distribuição das Variáveis-Chave por Perfil de Cliente (K={k_ideal})", fontsize=15, y=1.01)
axes = axes.flatten()

for i, feature in enumerate(features_cluster):
    sns.boxplot(
        data=df_acme, 
        x='Perfil_Cliente', 
        y=feature, 
        order=ordem_perfis,
        palette=cores_perfil, 
        ax=axes[i]
    )
    axes[i].set_title(f'Distribuição de: {feature}', fontsize=12)
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# Exportação final
ARQUIVO_CLUSTERIZADO = "base_acme_co_clusterizada.csv"
df_acme.to_csv(ARQUIVO_CLUSTERIZADO, index=False)
print(f"\nBase clusterizada salva com sucesso em: '{ARQUIVO_CLUSTERIZADO}'")