# main.py
import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n>>> INICIANDO: {script_path}")
    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True, check=False)
    if result.returncode != 0:
        print(f"\n ERRO: O script {script_path} falhou com código de saída {result.returncode}. Parando o pipeline.")
        sys.exit(result.returncode)
    print(f"\n SUCESSO: {script_path} concluído.")

if __name__ == "__main__":
    scripts = [
        "src/data_collection.py",
        "src/data_processing.py",
        "src/feature_engineering.py",
        "src/model.py"
    ]

    for s in scripts:
        if not os.path.exists(s):
            print(f"Aviso: script {s} não encontrado. Verifique os nomes.")
    print("\nINICIANDO PIPELINE: Coleta -> Processamento -> Features -> Treino")
    for s in scripts:
        run_script(s)
    print("\nPIPELINE FINALIZADO COM SUCESSO!")
 