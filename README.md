# Projeto de Previsão de Churn de Clientes de Cartão de Crédito

**Autor:** Vinícius Ramos  
**Data:** 06 de Outubro de 2025

## 1. Resumo do Projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de prever a probabilidade de um cliente de cartão de crédito encerrar seus serviços com o banco (fenômeno conhecido como *churn*).

Através da análise de dados demográficos e comportamentais dos clientes, construímos e otimizamos um modelo de classificação que alcançou **84% de recall**, o que significa que ele é capaz de identificar corretamente 84 de cada 100 clientes que estão prestes a sair. Isso permite que o banco tome ações de retenção proativas e direcionadas, reduzindo perdas de receita e fortalecendo o relacionamento com seus clientes.

## 2. O Conjunto de Dados

O dataset utilizado foi o `BankChurners.csv`, um conjunto de dados público que contém informações de mais de 10.000 clientes de um portfólio de cartão de crédito.

As principais informações incluem:
- **Dados Demográficos:** Idade, gênero, estado civil, nível de escolaridade e faixa de renda.
- **Dados do Produto:** Categoria do cartão (Blue, Silver, Gold, etc.), limite de crédito e meses como cliente.
- **Métricas de Comportamento:** Saldo rotativo, valor total de transações, número de transações, meses de inatividade e taxa de utilização do limite.
- **Variável Alvo:** `Attrition_Flag`, que indica se o cliente é existente (`Existing Customer`) ou se saiu (`Attrited Customer`).

## 3. Metodologia Aplicada

O projeto foi desenvolvido seguindo o ciclo de vida padrão de um projeto de ciência de dados:

#### a) Análise Exploratória de Dados (EDA)
Investigamos os dados para extrair insights iniciais. A principal descoberta foi que clientes que apresentam **queda abrupta no valor (`Total_Trans_Amt`) e na quantidade (`Total_Trans_Ct`) de transações** são os mais propensos a dar churn. A inatividade nos meses anteriores também se mostrou um forte indicador de risco.

#### b) Pré-processamento
Os dados foram preparados para a modelagem através das seguintes etapas:
1.  **Limpeza:** Remoção de colunas irrelevantes.
2.  **Engenharia de Features:** Conversão da variável alvo (`Attrition_Flag`) para um formato numérico (`Churn`: 1 ou 0).
3.  **One-Hot Encoding:** Transformação de todas as variáveis categóricas (como `Gender`, `Marital_Status`) em formato numérico para que o modelo pudesse processá-las.
4.  **Divisão dos Dados:** Separação do dataset em 80% para treino e 20% para teste, utilizando a técnica de **estratificação** (`stratify`) para garantir que a proporção de churn (16.1%) fosse mantida em ambos os conjuntos.

#### c) Modelagem e Otimização
1.  **Modelo Base:** Um primeiro modelo `RandomForestClassifier` foi treinado utilizando o parâmetro `class_weight='balanced'` para lidar com o desbalanceamento dos dados. Este modelo alcançou um bom resultado inicial, com 74% de recall.
2.  **Otimização de Hiperparâmetros:** Para refinar o modelo, utilizamos a técnica `GridSearchCV`. Testamos sistematicamente diversas combinações de hiperparâmetros (como `n_estimators`, `max_depth`, etc.), com o objetivo de **maximizar a métrica de recall**.
3.  **Modelo Final:** O modelo otimizado pelo `GridSearchCV` apresentou um desempenho superior, elevando o recall para 84%.

## 4. Como Executar o Projeto

#### Pré-requisitos
- Python 3.7+
- Gerenciador de pacotes pip

#### Instalação
Clone o repositório e instale as dependências necessárias:
```bash
# Clone este repositório (exemplo)
# git clone https://github.com/seu-usuario/seu-repositorio.git
# cd seu-repositorio

# Instale as bibliotecas
pip install pandas matplotlib seaborn scikit-learn
```

## 5. Resultados

A otimização do modelo resultou em uma melhora significativa na capacidade de identificar clientes em risco de churn, conforme demonstrado pela comparação dos relatórios de classificação para a classe `Saiu (1)`:

| Métrica   | Modelo Base | Modelo Refinado | Impacto                                                              |
| :-------- | :---------- | :-------------- | :------------------------------------------------------------------- |
| **Recall**  | 74%         | **84%**         | **+10%**: Identifica 10% a mais dos clientes que realmente saem.       |
| **Precision** | 94%         | **84%**         | **-10%**: Um leve aumento nos "falsos positivos", uma troca vantajosa. |
| **F1-Score**| 83%         | **84%**         | **+1%**: Melhor equilíbrio geral entre precisão e recall.            |

O modelo final (`best_model`) representa um ativo valioso, capaz de gerar listas de clientes de alto risco com alta acurácia e cobertura.

## 6. Próximos Passos
Como sugestões para trabalhos futuros, o projeto pode ser estendido das seguintes formas:
- **Análise de Features:** Realizar uma análise aprofundada da importância das features (`feature_importance_`) do modelo final para entender quais fatores mais influenciam o churn.
- **Deployment:** Implementar o modelo em um ambiente de produção através de uma API (ex: com Flask ou FastAPI) para realizar previsões em tempo real.
- **Outros Modelos:** Testar outros algoritmos de classificação, como Gradient Boosting (XGBoost, LightGBM), que podem oferecer ganhos adicionais de performance.
