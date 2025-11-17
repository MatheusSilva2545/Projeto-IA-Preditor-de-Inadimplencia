# src/data_processing.py
"""
Processamento básico do CSV de empréstimos -> normalização de colunas, tipos e limpeza.
Produz: data/processed/loan_clean.csv
"""
import pandas as pd
import os
import numpy as np

RAW_PATH = 'data/raw/Loan_default.csv'
PROCESSED_DIR = 'data/processed'
os.makedirs(PROCESSED_DIR, exist_ok=True)
OUT_PATH = os.path.join(PROCESSED_DIR, 'loan_clean.csv')

# Mapeamentos de possíveis nomes de colunas no CSV
COL_MAP_CANDIDATES = {
    'renda': ['renda', 'annual_income', 'income', 'income_annual', 'monthly_income'],
    'idade': ['idade', 'age', 'years'],
    'score': ['score', 'credit_score', 'fico', 'credit_score_value'],
    'valor': ['loan_amount', 'valor', 'loan_value', 'amount'],
    'meses_em_atraso': ['months_delayed', 'months_in_arrears', 'num_late_payments', 'months_late'],
    'target': ['default', 'delinquent', 'is_default', 'target']
}

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def process():
    if not os.path.exists(RAW_PATH):
        print(f"Arquivo bruto não encontrado em {RAW_PATH}. Rode src/data_collection.py primeiro.")
        return False

    df = pd.read_csv(RAW_PATH)
    df_columns_lower = [c.lower() for c in df.columns]

    # construir novo df com colunas padronizadas
    mapped = {}
    for std, candidates in COL_MAP_CANDIDATES.items():
        col = None
        for cand in candidates:
            if cand in df.columns:
                col = cand
                break
            # try lowercase match
            for orig in df.columns:
                if orig.lower() == cand.lower():
                    col = orig
                    break
            if col:
                break
        mapped[std] = col

    print("Mapeamento detectado (None = não encontrado):")
    for k,v in mapped.items():
        print(f"  {k}: {v}")

    # verificar colunas essenciais
    essentials = ['renda','idade','score','valor','meses_em_atraso','target']
    missing = [k for k in essentials if mapped.get(k) is None]
    if missing:
        print("Aviso: as colunas essenciais não foram todas encontradas:", missing)
        print("Se o seu CSV usa nomes diferentes, ajuste os mapeamentos em src/data_processing.py")
        # não aborta, tenta prosseguir com o que tiver

    # construir dataframe padronizado
    df2 = pd.DataFrame()
    for std in ['renda','idade','score','valor','meses_em_atraso','target']:
        colname = mapped.get(std)
        if colname is not None:
            df2[std] = df[colname]
        else:
            # se faltar, preencher com NaNs
            df2[std] = np.nan

    # Tipos e limpeza
    # renda e valor -> numérico (R$)
    for c in ['renda','valor','score','meses_em_atraso','idade']:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')

    # target -> binário 0/1
    if 'target' in df2.columns:
        df2['target'] = df2['target'].map(lambda x: 1 if str(x).strip() in ['1','True','true','YES','yes','Y','y'] else (0 if str(x).strip() in ['0','False','false','NO','no','N','n'] else x))
        df2['target'] = pd.to_numeric(df2['target'], errors='coerce').fillna(0).astype(int)
    else:
        # se não houver target, cria um proxy (meses_em_atraso >=3)
        if 'meses_em_atraso' in df2.columns:
            df2['target'] = (df2['meses_em_atraso'] >= 3).astype(int)
            print("Target não encontrado: criando proxy target = meses_em_atraso >= 3")
        else:
            df2['target'] = 0
            print("Nenhuma coluna de target ou meses em atraso encontrada. Target preenchido com zeros.")

    # preencher nulos razoavelmente
    df2['renda'] = df2['renda'].fillna(df2['renda'].median() if not df2['renda'].isna().all() else 0)
    df2['idade'] = df2['idade'].fillna(df2['idade'].median() if not df2['idade'].isna().all() else 25)
    df2['score'] = df2['score'].fillna(df2['score'].median() if not df2['score'].isna().all() else 600)
    df2['valor'] = df2['valor'].fillna(df2['valor'].median() if not df2['valor'].isna().all() else 10000)
    df2['meses_em_atraso'] = df2['meses_em_atraso'].fillna(0)

    df2.to_csv(OUT_PATH, index=False)
    print(f"Dados processados salvos em: {OUT_PATH}")
    print("Amostra:")
    print(df2.head())
    return True

if __name__ == "__main__":
    process()
