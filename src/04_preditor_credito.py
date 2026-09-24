'''
Arquivo.....: 04_score_confianca.py
Data........: 23/09/2026
Autor.......: Vladmir B. da Cruz
Contato.....: vladcruz@gmail.com
Descrição...: Rede Neural MLP para cálculo de probabilidade e Score de Confiança ponderado,
              integrado com Explicabilidade Local (XAI via LIME) para novos proponentes.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import warnings

# Silenciando warnings para manter a saída limpa
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração visual do Seaborn
sns.set_theme(style="whitegrid")

# ==========================================
# PARTE A: CARREGAMENTO DOS DADOS PARA TREINO
# ==========================================
ARQUIVO_CLUSTERIZADO = "base_acme_co_clusterizada.csv"
print(f"Carregando a base de conhecimento ({ARQUIVO_CLUSTERIZADO})...\n")

try:
    df_acme = pd.read_csv(ARQUIVO_CLUSTERIZADO)
except FileNotFoundError:
    print(f"Erro: O arquivo '{ARQUIVO_CLUSTERIZADO}' não foi encontrado. Execute os scripts anteriores primeiro.")
    exit()

features = [
    'Renda_Comprovada', 
    'Tempo_Cliente_Meses', 
    'Tempo_Pagamento_Dias', 
    'Comprometimento_Renda_Pct', 
    'Valor_Solicitado'
]

X = df_acme[features]
y = df_acme['Cluster']

# Mapeamento dinâmico das classes e perfis criados no Script 2
mapeamento_perfis = df_acme[['Cluster', 'Perfil_Cliente']].drop_duplicates().sort_values('Cluster')
nomes_classes = mapeamento_perfis['Perfil_Cliente'].tolist()
k_total = len(nomes_classes)

print(f"Clusters detectados no ambiente: K = {k_total}")
for cid, cnome in zip(mapeamento_perfis['Cluster'], nomes_classes):
    print(f"  -> Nível {cid}: {cnome}")
print()

# Divisão em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização (Z-score): Fundamental para a convergência estável da Rede Neural MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# PARTE B: TREINANDO O MOTOR DE CRÉDITO (REDE NEURAL MLP)
# ==========================================
print("Treinando o Motor de Crédito (Rede Neural MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=1000, 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

acuracia_teste = mlp.score(X_test_scaled, y_test) * 100
print(f"Acurácia do modelo em dados de teste: {acuracia_teste:.2f}%\n")

# ==========================================
# PARTE C: NOVOS CLIENTES E LÓGICA DE NEGÓCIO (SCORE DINÂMICO)
# ==========================================
print("=== SIMULANDO NOVOS PEDIDOS DE EMPRÉSTIMO ===")

novos_clientes = pd.DataFrame({
    'Nome': ['Ana (A Executiva)', 'Bruno (O Endividado)', 'Carlos (Trabalhador Médio)'],
    'Renda_Comprovada': [22000, 3500, 8000],
    'Tempo_Cliente_Meses': [60, 5, 24],
    'Tempo_Pagamento_Dias': [-3, 15, 2],
    'Comprometimento_Renda_Pct': [15, 65, 35],
    'Valor_Solicitado': [10000, 20000, 5000]
})

X_novos = novos_clientes[features]
X_novos_scaled = scaler.transform(X_novos)

# Inferência das probabilidades e do perfil dominante
probabilidades = mlp.predict_proba(X_novos_scaled)
perfis_preditos = np.argmax(probabilidades, axis=1)

# Ponderação do Fator de Confiança:
pesos_confianca = np.linspace(100, 0, k_total).round(1)

fator_confianca_lista = []
decisao_lista = []

for i in range(len(novos_clientes)):
    score = np.dot(probabilidades[i], pesos_confianca)
    fator_confianca_lista.append(score)
    
    # Políticas de corte de crédito
    if score >= 70:
        decisao_lista.append("APROVADO AUTOMATICAMENTE")
    elif score <= 40:
        decisao_lista.append("RECUSADO (ALTO RISCO)")
    else:
        decisao_lista.append("ENCAMINHADO PARA ANÁLISE HUMANA")

novos_clientes['Score'] = fator_confianca_lista
novos_clientes['Decisao'] = decisao_lista
novos_clientes['Perfil_Base'] = [nomes_classes[p] for p in perfis_preditos]

# ==========================================
# PARTE D: CONFIGURANDO O LIME PARA REDES NEURAIS
# ==========================================
# Função wrapper: o LIME perturba dados na escala original e a função aplica o Scaler antes do MLP
def predict_fn_com_scaler(X_raw):
    X_scaled_temp = scaler.transform(X_raw)
    return mlp.predict_proba(X_scaled_temp)

explainer_mlp = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values, 
    feature_names=features,
    class_names=nomes_classes,
    mode='classification',
    random_state=42
)

# ==========================================
# PARTE E: PAINEL DE DECISÃO DE CRÉDITO (XAI + DASHBOARD)
# ==========================================
print("Gerando o Dashboard Final do Analista de Crédito...")

fig, axes = plt.subplots(1, 3, figsize=(20, 10)) 
fig.suptitle("ACME Co. - Painel de Decisão de Crédito Explicável (XAI + Deep Learning)", fontsize=18, fontweight='bold')

for i in range(len(novos_clientes)):
    nome = novos_clientes['Nome'].iloc[i]
    score = novos_clientes['Score'].iloc[i]
    decisao = novos_clientes['Decisao'].iloc[i]
    perfil = novos_clientes['Perfil_Base'].iloc[i]
    
    # Atributos brutos da proposta
    renda = novos_clientes['Renda_Comprovada'].iloc[i]
    solicitado = novos_clientes['Valor_Solicitado'].iloc[i]
    tempo = novos_clientes['Tempo_Cliente_Meses'].iloc[i]
    pgto = novos_clientes['Tempo_Pagamento_Dias'].iloc[i]
    compr = novos_clientes['Comprometimento_Renda_Pct'].iloc[i]
    
    # Explicação local para a proposta específica
    exp = explainer_mlp.explain_instance(
        data_row=X_novos.iloc[i].values, 
        predict_fn=predict_fn_com_scaler, 
        num_features=len(features), 
        top_labels=1
    )
    
    classe_explicada = exp.available_labels()[0]
    explicacao_lista = exp.as_list(label=classe_explicada)
    
    regras = [item[0] for item in explicacao_lista]
    pesos = [item[1] for item in explicacao_lista]
    regras.reverse()
    pesos.reverse()
    
    # Cores: Verde fortalece a classificação, Vermelho atenua
    cores = ['#2ca02c' if p > 0 else '#d62728' for p in pesos]
    
    ax = axes[i]
    barras = ax.barh(regras, pesos, color=cores, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    
    # Card formatado com dados da proposta
    titulo = (
        f"Cliente: {nome}\n"
        f"Classificação: {perfil}  |  Score: {score:.1f}/100\n"
        f"----------------------------------------------------\n"
        f"Renda: R$ {renda:,.2f}  |  Crédito Solicitado: R$ {solicitado:,.2f}\n"
        f"Comprometimento: {compr}%  |  Média de Pagamento: {pgto} dias\n"
        f"Tempo de Relacionamento: {tempo} meses"
    )
    
    ax.set_title(titulo, fontsize=11, loc='left', pad=45)
    
    # Carimbo visual do status do crédito
    cor_decisao = '#2ca02c' if 'APROVADO' in decisao else ('#d62728' if 'RECUSADO' in decisao else '#ff7f0e')
    ax.text(
        0.5, 1.05, decisao, 
        transform=ax.transAxes, 
        fontsize=12.5, 
        fontweight='bold', 
        color=cor_decisao, 
        ha='center', 
        va='bottom', 
        bbox=dict(facecolor='white', edgecolor=cor_decisao, boxstyle='round,pad=0.35', lw=1.5)
    )
    
    # Rótulo numérico das barras do LIME
    for barra in barras:
        largura = barra.get_width()
        pos_x = largura + 0.01 if largura >= 0 else largura - 0.01
        alinhamento = 'left' if largura >= 0 else 'right'
        ax.text(
            pos_x, barra.get_y() + barra.get_height() / 2, f'{largura:+.2f}', 
            va='center', ha=alinhamento, fontsize=10, fontweight='bold'
        )

# Reserva o espaço superior (rect) para os cabeçalhos das propostas sem sobreposição
plt.tight_layout(w_pad=3.0, rect=[0, 0, 1, 0.75])
plt.show()

print("\nPipeline de Crédito e Explicabilidade concluído com sucesso!")