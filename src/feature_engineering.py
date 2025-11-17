# src/feature_engineering.py
"""
Criação de features para o modelo de inadimplência.
Entrada: data/processed/loan_clean.csv
Saída: data/features/loan_features.csv
"""
import pandas as pd
import os
import numpy as np

PROCESSED_PATH = 'data/processed/loan_clean.csv'
FEATURE_DIR = 'data/features'
os.makedirs(FEATURE_DIR, exist_ok=True)
OUT_PATH = os.path.join(FEATURE_DIR, 'loan_features.csv')

def make_features():
    if not os.path.exists(PROCESSED_PATH):
        print(f"Arquivo processado não encontrado em {PROCESSED_PATH}. Rode src/data_processing.py primeiro.")
        return False

    df = pd.read_csv(PROCESSED_PATH)

    # renomear colunas esperadas caso necessário (garantia)
    expected = ['renda','idade','score','valor','meses_em_atraso','target']
    for c in expected:
        if c not in df.columns:
            df[c] = 0

    # Feature 1: proporção do valor do empréstimo em relação à renda anual
    # cuidado: renda pode ser mensal ou anual; assumimos renda mensal -> transformar para anual
    renda_annual = df['renda'] * 12
    df['loan_to_income'] = df['valor'] / (renda_annual.replace(0, np.nan))
    df['loan_to_income'] = df['loan_to_income'].fillna(df['valor'] / (df['renda'].median()*12 + 1e-6))

    # Feature 2: parcela estimada (hipótese simplificada: 5 anos)
    df['estimated_monthly_payment'] = df['valor'] / (5*12)

    # Feature 3: porcentagem da renda comprometida
    df['pct_income_commitment'] = df['estimated_monthly_payment'] / (df['renda'].replace(0, np.nan))
    df['pct_income_commitment'] = df['pct_income_commitment'].fillna(df['estimated_monthly_payment'] / (df['renda'].median()+1e-6))

    # Feature 4: idade categórica
    df['age_bucket'] = pd.cut(df['idade'], bins=[0,24,34,44,54,100], labels=['<=24','25-34','35-44','45-54','55+'])

    # Feature 5: score bucket
    df['score_bucket'] = pd.cut(df['score'], bins=[0,550,650,750,850,1000], labels=['baixo','medio-baixo','medio','alto','excelente'])

    # Feature 6: flag atraso
    df['overdue_flag'] = (df['meses_em_atraso'] > 0).astype(int)
    df['serious_arrears'] = (df['meses_em_atraso'] >= 3).astype(int)

    # Selecionar colunas finais para o modelo
    features = [
        'renda','idade','score','valor','meses_em_atraso',
        'loan_to_income','estimated_monthly_payment','pct_income_commitment',
        'overdue_flag','serious_arrears','age_bucket','score_bucket','target'
    ]
    df_final = df[features].copy()

    # salvar
    df_final.to_csv(OUT_PATH, index=False)
    print(f"Features salvas em: {OUT_PATH}")
    print(df_final.head())
    return True

if __name__ == "__main__":
    make_features()
