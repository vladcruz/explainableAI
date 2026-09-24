# Explainable AI (XAI)

## Da resposta convincente à decisão defensável

Este repositório nasceu de uma pergunta simples: **como confiar em uma decisão apoiada por Inteligência Artificial sem transformar uma explicação bonita em prova de que o sistema está correto?**

A proposta é explorar Explainable AI — ou XAI — de maneira acessível, técnica e conectada ao mundo real. Em vez de tratar explicabilidade como um gráfico acrescentado ao final do projeto, o conteúdo relaciona modelos, dados, evidências, pessoas, processos, riscos e responsabilidades.

O material combina duas partes:

- Um guia conceitual sobre explicabilidade, interpretabilidade, transparência, rastreabilidade e governança de IA.
- Quatro scripts Python na pasta [`src`](./src), que transformam os conceitos discutidos em demonstrações práticas.

> Uma resposta convincente não é necessariamente uma resposta confiável. Explicar é tornar possível investigar, questionar e reconstruir.

## O que você encontra aqui

O conteúdo percorre temas como:

- A diferença entre fluência, plausibilidade, evidência e decisão.
- Interpretabilidade, explicabilidade, transparência e rastreabilidade.
- Explicações globais e locais.
- Importância e efeito das variáveis.
- SHAP, LIME e explicações contrafactuais.
- Modelos naturalmente interpretáveis.
- Explicabilidade em redes neurais e IA generativa.
- Avaliação da fidelidade, estabilidade e utilidade das explicações.
- Viés, justiça, privacidade e segurança.
- Supervisão humana, contestação e prestação de contas.
- Governança contínua de IA.
- LGPD, EU AI Act, NIST AI RMF e ISO/IEC 42001.
- XAI como capacidade corporativa, e não apenas como ferramenta técnica.

A ideia não é oferecer uma “explicação definitiva” para qualquer modelo. O objetivo é mostrar que cada método responde a uma pergunta diferente e possui limites que precisam ser conhecidos.

## A parte prática

A pasta [`src`](./src) contém quatro scripts Python preparados para exemplificar os conceitos apresentados no material.

Eles funcionam como uma ponte entre teoria e prática: permitem observar como uma previsão pode ser investigada, como diferentes explicadores produzem perspectivas distintas e por que uma saída visualmente clara ainda precisa ser validada.

Os exemplos devem ser lidos como demonstrações técnicas. Resultados, atributos importantes e explicações dependem do modelo, dos dados, do conjunto de referência, dos parâmetros e do contexto de uso.

```text
.
├── LICENSE
├── README.md
├── Do 42 à Confiança - Explainable AI e Governança.pdf
├── src/
│   ├── 01_gera_base.py
│   ├── 02_agrupa_base.py
│   ├── 03_confere_perfil.py
│   ├── 04_preditor_credito.py
│   └── requirements.txt
└── [materiais futuros]
```

## Como executar

### Pré-requisitos

- Python 3.10 ou superior.
- `pip` para instalação das dependências (use o requirements.txt dentro do src/).
- Ambiente virtual recomendado.

### Preparação do ambiente

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Caso o repositório tenha um arquivo `requirements.txt`, instale as dependências com:

```bash
pip install -r requirements.txt
```

### Execução dos exemplos

Execute cada arquivo separadamente pela IDE ou pelo terminal. Essa opção é recomendada para acompanhar as etapas, observar as saídas e comparar os resultados com as explicações do guia.

## Como interpretar os resultados

Ao analisar uma explicação produzida pelos scripts, vale observar:

- **Pergunta:** o método está explicando o modelo inteiro ou apenas uma previsão?
- **Referência:** qual conjunto ou valor de base foi usado para comparação?
- **Escala:** a saída representa probabilidade, log-odds, classe ou valor bruto?
- **Fidelidade:** a explicação acompanha o comportamento real do modelo?
- **Estabilidade:** pequenas mudanças produzem explicações semelhantes?
- **Causalidade:** o método descreve associações do modelo, não necessariamente causas no mundo real.
- **Ação:** a informação ajuda a investigar, revisar ou contestar o resultado?

SHAP, LIME, importância por permutação, PDP, ICE e contrafactuais não são versões concorrentes da mesma resposta. Cada abordagem observa o sistema por um ângulo diferente.

## Ideia central

Um modelo produz estimativas. A política da organização estabelece critérios. Regras tratam condições explícitas. Pessoas e instâncias autorizadas assumem decisões e responsabilidades.

Misturar essas funções cria a impressão de que o modelo “decidiu sozinho”. Separá-las torna o processo mais compreensível, auditável e governável.

**A IA estima. A organização decide. Pessoas respondem.**

## Limites do projeto

Este repositório tem finalidade educacional e de experimentação.

- As demonstrações não devem ser utilizadas diretamente em produção.
- Uma explicação não comprova que o modelo está correto, justo ou seguro.
- Os resultados não substituem validação estatística, conhecimento do domínio, avaliação de impacto ou revisão especializada.
- Questões jurídicas e regulatórias devem ser avaliadas conforme a jurisdição, o setor e o caso de uso.
- Antes de usar XAI em decisões reais, é necessário revisar dados, métricas, grupos afetados, segurança, privacidade, supervisão e mecanismos de contestação.

## Tecnologias relacionadas

Dependendo do tipo de modelo e da pergunta investigada, o ecossistema pode incluir:

- **Modelagem e inspeção:** scikit-learn, statsmodels e InterpretML.
- **Explicações locais e globais:** SHAP, LIME e Alibi Explain.
- **Contrafactuais:** DiCE, Alibi Explain e CARLA.
- **Redes neurais e visão:** Captum, tf-explain, Grad-CAM e Xplique.
- **Justiça algorítmica:** Fairlearn e AI Fairness 360.
- **Registro e MLOps:** MLflow e registries de modelos.
- **Monitoramento:** Evidently AI, WhyLabs, Arize e Fiddler.
- **Governança:** NIST AI RMF, ISO/IEC 42001, inventários de IA e plataformas de GRC.

A ferramenta vem depois da pergunta. Um método sofisticado não compensa um problema mal definido, dados inadequados ou ausência de responsabilidade.

## Para continuar explorando

Alguns pontos de partida importantes:

- [NIST — Four Principles of Explainable Artificial Intelligence](https://nvlpubs.nist.gov/nistpubs/ir/2021/NIST.IR.8312.pdf)
- [NIST — AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Interpretable Machine Learning — Christoph Molnar](https://christophm.github.io/interpretable-ml-book/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME — artigo original](https://arxiv.org/abs/1602.04938)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [Lei Geral de Proteção de Dados — LGPD](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [Regulamento Europeu de Inteligência Artificial](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)

## Autoria e apoio

**Autor, curador e responsável editorial:** Vladmir Cruz  
**Apoio à pesquisa, estruturação e revisão textual:** Perplexity AI

A seleção das fontes, a revisão técnica, os exemplos autorais e a validação final do conteúdo são de responsabilidade do autor.

## Licenças

Este repositório reúne conteúdo educacional e código-fonte; por isso, cada parte pode receber uma licença adequada à sua natureza:

- **Textos, guia e documentação:** [Creative Commons Atribuição 4.0 Internacional — CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.pt-br).
- **Scripts Python em `src`:** licença de software indicada no arquivo [`LICENSE`](./LICENSE), quando presente.

Para uma publicação aberta e permissiva do código, a licença MIT é uma opção simples: permite uso, cópia, modificação e distribuição, desde que o aviso de copyright e a licença sejam preservados.

Materiais de terceiros, marcas, bibliotecas, bases de dados e documentos citados permanecem sujeitos às licenças e aos direitos de seus respectivos titulares.

### Atribuição sugerida

> CRUZ, Vladmir. *Explainable AI (XAI): da resposta convincente à decisão defensável*. 2026. Material educacional e exemplos em Python. Apoio à pesquisa, estruturação e revisão textual: Perplexity AI. Licença do conteúdo: CC BY 4.0.

## Contribuições

Sugestões, correções e novas perspectivas são bem-vindas. Ao abrir uma *issue* ou propor uma alteração, procure informar:

- Qual conceito, exemplo ou trecho está sendo discutido.
- Qual comportamento foi observado.
- Como reproduzir o resultado, quando houver código envolvido.
- Quais fontes ou evidências sustentam a proposta.

XAI melhora quando as explicações também podem ser questionadas.

---

*Se este material ajudar a olhar para uma previsão com um pouco mais de curiosidade — e um pouco menos de confiança automática — ele já terá cumprido seu papel.*
