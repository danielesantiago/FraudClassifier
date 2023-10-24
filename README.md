# Detecção de Transações Fraudulentas

![image](https://github.com/danielesantiago/FraudClassifier/assets/64613885/2f879988-ada6-48f0-bdfe-b5557308899e)



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

## 📊 Análise do Modelo Atual

A eficácia do nosso modelo de detecção de fraude é vital para a saúde financeira da empresa. A empresa ganha 10% do valor de um pagamento corretamente aprovado, mas sofre uma perda completa, ou seja, 100%, em caso de uma transação fraudulenta. Portanto, otimizar a Taxa de Fraude e a Taxa de Aprovação é fundamental.

Os dados atuais apontam para algumas áreas de preocupação:

![image](https://github.com/danielesantiago/FraudClassifier/assets/64613885/4887e641-a6d6-4256-bdc7-3ea26a7849d2)


Ao observar o gráfico, notamos um problema significativo: as classes estão notavelmente sobrepostas. Isso indica que nosso modelo tem dificuldades em distinguir entre transações legítimas e fraudulentas. Idealmente, gostaríamos de ver uma separação mais clara entre as duas classes, o que indicaria que o modelo pode identificar características distintas associadas a cada tipo de transação. A sobreposição sugere que muitas transações legítimas e fraudulentas têm características semelhantes, tornando a tarefa de classificação mais desafiadora.

- **Financeiro**:
  - **Perdas com fraudes**: $25,353.320
  - **Receitas**: $80,329.995
  - **Lucro líquido**: $54,976.675
  - **Razão de lucro**: 68.44%.

- **Desempenho do Modelo**:
  - **Taxa de Fraude**: 2%
  - **Taxa de Aprovação**: 74%
  - **Log Loss**: 8.2526
  - **ROC-AUC**: 0.7193
 
A falta de distinção clara entre as classes, como ilustrado na imagem, combinada com as métricas de desempenho fornecidas, sugere a necessidade de uma revisão e possivelmente uma reformulação do modelo. Isso pode envolver considerar outras features, aplicar técnicas de balanceamento de classes ou experimentar diferentes algoritmos de classificação.


## 🛠 Pré-processamento 
O pré-processamento de dados é uma etapa crucial em projetos de machine learning, e para garantir a eficácia e reprodutibilidade do nosso processo, utilizamos o **Pipeline do Scikit-learn**. O uso de um pipeline assegura que as transformações aplicadas aos dados de treinamento sejam reproduzidas de forma idêntica nos dados de teste, eliminando potenciais erros e inconsistências.

_Considerações_:
1. A coluna valor_compra refere-se ao valor da compra e está em uma única unidade (ex: Dólar).
2. Não há custo extra de fraude além do mencionado.
3. Nenhuma das colunas inseridas no modelo causará data leakage; ou seja, todos esses dados são calculados/recebidos antes que o evento "Fraude" ocorra.
   
_Etapas do Pré-processamento no Pipeline_:
1. Exclusão de Colunas:
* score_fraude_modelo: Modelo baseline que não deve ser utilizado.
* data_compra: Para prevenir a degradação do modelo com o tempo.
* produto: Devido à alta cardinalidade (mais de 8 mil categorias).

2. Tratamento de Categorias:
* Manter as 1000 categorias em categoria_produto que correspondem a 80% das fraudes.
* Limitar país para "BR", "AR" (que compõem mais de 90% da distribuição) e "outros".
* Target encoding em categoria_produto devido à alta cardinalidade.
* One hot encoding nas demais variáveis categóricas.

3. Tratamento de Valores Nulos:
* Preencher os nulos de score com a mediana, visto que não seguem uma distribuição normal.
* Criar uma feature is_null indicando quais valores de entrega_doc_2 são nulos.
* Considerar os nulos de entrega_doc_2 como 0, indicando "não entregue".
  
## 🤖 MLFlow

MLFlow é uma plataforma aberta de gerenciamento do ciclo de vida de machine learning. Ele oferece ferramentas para rastrear experimentos, empacotar código em formatos reprodutíveis e compartilhar e implantar modelos. Uma das principais vantagens do MLFlow é a sua capacidade de registrar métricas, parâmetros e artefatos, facilitando a comparação entre diferentes versões de modelos e a reprodução de experimentos.

Para o nosso projeto, utilizamos o MLFlow como ferramenta de rastreamento. As métricas priorizadas para avaliação e comparação dos modelos foram:
- Taxa de aprovação
- Lucro gerado pelo modelo atual
- ROC-AUC (área sob a curva característica de operação do receptor)
- Razão de lucro
  
Dessa maneira, foi possível monitorar e otimizar o desempenho do nosso modelo de forma eficiente, garantindo resultados mais robustos e transparentes. A visualização das métricas e experimentos do projeto podem ser visualizadas no dashboard MLFlow:

![image](https://github.com/danielesantiago/FraudClassifier/assets/64613885/d4424e41-4153-4331-90b1-27266dc8c965)



## 📈 Modelo Treinado

Ao avaliar o desempenho do Modelo Atual em comparação com o Modelo Treinado, é evidente que o último apresenta melhorias significativas não apenas em métricas de desempenho, mas também no impacto financeiro.

![image](https://github.com/danielesantiago/FraudClassifier/assets/64613885/b5a946df-0d58-4756-965a-cbaccaae2c5a)


**Análise Visual da Imagem**:
Ao examinar o gráfico, vemos uma distinção mais clara entre as transações legítimas e as fraudulentas no Modelo Treinado. Essa separação melhor definida sugere que o modelo é mais capaz de identificar as características distintivas das transações, resultando em classificações mais precisas.

**Desempenho Financeiro e Métricas**:
- **Threshold ótimo**: 61
- **Perdas com fraudes**: $27,007.200
- **Receitas**: $95,198.968
- **Lucro líquido**: $68,191.768
- **ROC-AUC**: 0.8512
- **Taxa de Fraude**: 0.02
- **Taxa de Aprovação**: 0.85
- **Razão Lucro/Receitas**: 72%

**Comparação entre Modelo Atual e Modelo Treinado**:

- **Taxa de fraude**: Mantém-se constante em 0.02 para ambos os modelos.
- **Taxa de aprovação**: O Modelo Treinado tem uma taxa de aprovação superior, 0.85 em comparação com 0.74 do Modelo Atual.
- **Razão Lucro/Receitas**: O Modelo Treinado mostra uma melhoria de 4%, passando de 68% no Modelo Atual para 72%.

## 📊 Análise Exploratória, SHAP e Testes de Hipóteses

Para aprofundar nosso entendimento sobre o comportamento do modelo, conduzimos uma análise exploratória detalhada, complementada pela análise SHAP. Essa análise SHAP nos permitiu destrinchar a relevância de cada variável e entender seu impacto nas previsões. Adicionalmente, realizamos testes de hipóteses para validar e solidificar nossas descobertas, garantindo que as observações são estatisticamente significativas. O detalhamento dessas análises pode ser acessado em nosso repositório: 
[Case Fraude no GitHub](https://github.com/danielesantiago/FraudClassifier/blob/master/Case%20Fraude.ipynb).

## 🚀 Reflexões Finais

Os avanços nas métricas de desempenho, combinados com uma análise financeira promissora, indicam que o Modelo Treinado é uma evolução notável em relação ao anterior. Ele se posiciona não apenas como uma ferramenta mais eficaz para identificar fraudes, mas também como um potencial catalisador para aumentar a rentabilidade.

## 🚧 Próximos Passos

O próximo marco neste percurso é levar o Modelo Treinado ao campo de batalha real: seu deployment. Isso permitirá que o modelo seja utilizado no cotidiano operacional, traduzindo suas promessas teóricas em benefícios tangíveis e imediatos para a empresa.

