# Detecção de Transações Fraudulentas

## 📌 Overview
O projeto visa conduzir uma análise exploratória dos dados e construir modelos de machine learning para detectar transações fraudulentas com alta precisão. Utilizamos técnicas avançadas de análise de dados, machine learning e balanceamento de dados para identificar padrões e anomalias.

## 💼 Business Understanding

A fraude em transações representa uma ameaça crescente. À medida que o uso de cartão de crédito aumenta, as fraudes também acompanham esse crescimento. Entender essa ameaça e suas métricas é fundamental para combater efetivamente tais atividades. 

**Tipos de Fraude:**
- Fraude de Cartão de Crédito
  - Offline
  - Online
- Fraude em Telecomunicações
- Fraude em Computadores
- Outros Tipos (Fraude de Falência, Fraude de Furto/Contrafação, etc.)

**Principais KPIs (Indicadores Chave de Desempenho) de Fraude:**
- Aceitação
- Desafios
- Negações
- Chargebacks
- Falsos Positivos

## 📊 Métricas de Business
A empresa ganha 10% do valor de um pagamento corretamente aprovado e perde 100% em caso de fraude. Monitoramos a Taxa de Fraude e a Taxa de Aprovação para avaliar a eficácia de nossos modelos.

## 🛠 Pré-processamento

Tomamos diversas ações para o pré-processamento dos dados, como exclusão de colunas que poderiam causar data leakage, tratamento de valores nulos e codificação de variáveis categóricas.

## 📈 Análise da Diferença em Lucro

Avaliamos a eficácia financeira do Modelo Atual em comparação com o Modelo Treinado e descobrimos que o Modelo Treinado tem o potencial de trazer ganhos financeiros significativos.

**Modelo Atual vs Modelo Treinado:**
- Taxa de fraude: 0.02 (ambos)
- Taxa de aprovação: 0.74 vs 0.85
- Razão Lucro/Receitas: 68% vs 72%

## 📊 Análise SHAP 

Além das análises anteriores, realizamos uma análise SHAP para entender a importância de cada feature no modelo e conduzimos testes de hipóteses para confirmar a eficácia de nossas descobertas.

## 🚀 Conclusão Geral

Com métricas de desempenho melhoradas e uma avaliação financeira favorável, a implementação do Modelo Treinado é uma escolha estratégica robusta. Ele não só aprimora a detecção e prevenção de fraudes, mas também promete ganhos financeiros significativos.

## Próximos Passos
A próxima etapa crucial deste projeto é o deployment do Modelo Treinado, permitindo que ele seja usado em um ambiente de produção para começar a oferecer seus benefícios imediatamente.

