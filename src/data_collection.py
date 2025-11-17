# src/data_collection.py
"""
Coleta de dados:
- copia Loan_default.csv para data/raw/ (se necessário)
- consulta ao Banco Central (SGS) para indicadores macroeconômicos (ex.: SELIC/IPCA)
- salva indicadores em data/raw/macros_bcb.csv
"""
import os
import shutil
import pandas as pd
import requests
from datetime import datetime

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
SRC_CANDIDATES = ["/mnt/data/Loan_default.csv", "Loan_default.csv", os.path.join(RAW_DIR, "Loan_default.csv")]
DEST = os.path.join(RAW_DIR, "Loan_default.csv")

def copy_local_csv():
    for p in SRC_CANDIDATES:
        if os.path.exists(p):
            if os.path.abspath(p) != os.path.abspath(DEST):
                try:
                    shutil.copyfile(p, DEST)
                    print(f"Arquivo copiado de {p} -> {DEST}")
                except Exception as e:
                    print(f"Aviso: não foi possível copiar {p} -> {e}")
            else:
                print(f"Arquivo já em {DEST}")
            return True
    print("Arquivo Loan_default.csv não encontrado em locais padrão. Coloque-o em data/raw/")
    return False

def fetch_bcb_series(series_id="432", start_date="2018-01-01", end_date=None):
    """
    Busca série do SGS (Banco Central). Ex.: 432 (SELIC meta) — ajuste se quiser outro indicador.
    Retorna DataFrame com colunas ['data','valor'].
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_id}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df['data'] = pd.to_datetime(df['data'], dayfirst=False)
            df['valor'] = pd.to_numeric(df['valor'].str.replace(',','.'), errors='coerce')
            return df[['data','valor']]
    except Exception as e:
        print(f"Aviso: falha ao buscar série {series_id} no BCB: {e}")
    return pd.DataFrame()

def save_macros():
    # tentar 2 séries (exemplo): SELIC meta (4389?) e IPCA (433?) -> se não souber a série exata, manter genérica
    # Observação: ajuste series_id conforme necessidade; este código tenta 1 e devolve o que conseguir.
    series_to_try = ["432", "433"]  # placeholders; 432/433 may correspond to different series; code tolera falha
    out = []
    for s in series_to_try:
        df = fetch_bcb_series(series_id=s, start_date="2018-01-01")
        if not df.empty:
            df['series_id'] = s
            out.append(df)
    if out:
        df_all = pd.concat(out, ignore_index=True)
        path = os.path.join(RAW_DIR, "macros_bcb.csv")
        df_all.to_csv(path, index=False)
        print(f"Indicadores macro salvos em: {path}")
    else:
        print("Nenhum indicador macro foi salvo (APIs falharam ou retornaram vazio).")

if __name__ == "__main__":
    copy_local_csv()
    save_macros()
