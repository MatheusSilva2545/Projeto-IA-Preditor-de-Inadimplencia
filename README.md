## Sistema Preditivo de Inadimplência Estudantil

## Objetivo do Projeto

Desenvolver um Produto Mínimo Viável (MVP) de Machine Learning para prever o risco de inadimplência estudantil em financiamentos educacionais (como FIES ou programas próprios de instituições).
O sistema foi projetado para gerar insights úteis, permitindo identificar alunos com maior probabilidade de atraso ou não pagamento.


## Escalabilidade e Coerência

O pipeline foi construído para garantir consistência total entre a base de dados, a análise exploratória, o treinamento do modelo e o dashboard final.

1. **Dataset:** Base estruturada com Feature Engineering, contendo variáveis financeiras e demográficas.
2. **Coerência:** O processamento dos dados, o modelo e o dashboard utilizam a mesma base final (`loan_features.csv`), assegurando integridade técnica durante todo o fluxo.


## Estrutura do Repositório

| Arquivo/Pasta                     | Descrição                                                             |
| --------------------------------- | --------------------------------------------------------------------- |
| `src/model.py`                    | Script responsável pelo treinamento do modelo de Regressão Logística. |
| `src/data_processing.py`          | Limpeza e padronização dos dados brutos.                              |
| `src/feature_engineering.py`      | Criação de novos atributos derivados.                                 |
| `dashboard.py`                    | Aplicação interativa em Streamlit para predição.                      |
| `main.py`                         | Pipeline completo que executa todas as etapas do projeto.             |
| `requirements.txt`                | Lista de dependências necessárias.                                    |
| `models/`                         | Contém o modelo treinado (`model_pipeline_noleak.pkl`).               |
| `data/features/loan_features.csv` | Base de dados final utilizada no treinamento.                         |


# Guia de Execução Completo (Passo a Passo)

A seguir, um guia detalhado para configurar o ambiente e executar o sistema.


## 1. Preparação do Ambiente

Pré-requisito: Python 3.8 ou superior.

1. Clone ou baixe o repositório.
2. Crie e ative o ambiente virtual:

```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```


## 2. Treinar o Modelo de Machine Learning

Com o ambiente virtual ativo:

```bash
python src/model.py
```

O script irá gerar:

* O modelo treinado: `models/model_pipeline_noleak.pkl`
* As métricas de avaliação: `models/metrics_summary_noleak.csv`


## 3. Executar o Dashboard Interativo

Com o ambiente virtual ativo, execute:

```bash
streamlit run dashboard.py
```

No terminal será exibido um link como:

```
http://localhost:8501
```

Basta abrir no navegador.


## Funcionalidades do Dashboard

* Inserção de informações do estudante:

  * renda
  * idade
  * score
  * valor do financiamento
  * meses em atraso

* Predição automática do risco:

  * 0 = Adimplente
  * 1 = Inadimplente

* Gráfico interativo mostrando os valores inseridos.

* Interface intuitiva para uso por analistas financeiros e acadêmicos.


## Diferencial Estratégico

O sistema foi desenvolvido com foco em:

* Escalabilidade e manutenção facilitada.
* Modelo interpretável (Regressão Logística).
* Facilidade de integração com sistemas existentes.
* Potencial expansão para incluir histórico de pagamentos, curso, tipo de financiamento e comportamento longitudinal.


## Autor

Matheus da Silva
Disciplina: Inteligência Artificial

