import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================
# 1) CARREGAR MODELO
# ============================

MODEL_PATH = "models/model_pipeline_noleak.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {e}")
    st.stop()

# ============================
# 2) CONFIGURAÇÃO
# ============================

st.set_page_config(page_title="Preditor de Inadimplência", page_icon="🎓", layout="wide")

st.title("🎓 Preditor de Inadimplência Estudantil")
st.write("Preencha os dados do aluno para calcular o risco.")

st.markdown("---")

# ============================
# 3) ENTRADAS DO USUÁRIO
# ============================

st.sidebar.header("📌 Insira os dados")

renda = st.sidebar.number_input("Renda Mensal (R$)", min_value=0, max_value=30000, value=2500)
idade = st.sidebar.number_input("Idade", min_value=16, max_value=80, value=22)
score = st.sidebar.number_input("Score de Crédito", min_value=0, max_value=1000, value=650)
valor = st.sidebar.number_input("Valor financiado (R$)", min_value=0, max_value=200000, value=30000)
meses = st.sidebar.number_input("Meses em atraso", min_value=0, max_value=24, value=0)

# ============================
# 4) CRIAÇÃO DAS FEATURES IGUAIS AO TREINO
# ============================

df = pd.DataFrame({
    "renda": [renda],
    "idade": [idade],
    "score": [score],
    "valor": [valor],
    "meses_atraso": [meses]
})

# Features derivadas
df["loan_to_income"] = (valor / renda) if renda > 0 else 0
df["estimated_monthly_payment"] = valor / 12
df["pct_income_commitment"] = (df["estimated_monthly_payment"] / renda) if renda > 0 else 0

# Buckets
df["age_bucket"] = pd.cut(
    df["idade"],
    bins=[0, 25, 40, 200],
    labels=["jovem", "adulto", "senior"]
)

df["score_bucket"] = pd.cut(
    df["score"],
    bins=[0, 400, 700, 1000],
    labels=["baixo", "medio", "alto"]
)

# One-Hot Encoding
df = pd.get_dummies(df)

# ============================
# 5) AJUSTAR COLUNAS PARA O MODELO
# ============================

# Pegamos as colunas EXATAMENTE como no modelo
model_cols = model.feature_names_in_

# Criar colunas faltantes
for col in model_cols:
    if col not in df.columns:
        df[col] = 0

# Garantir ordem exata
df = df[model_cols]

# ============================
# 6) PREDIÇÃO
# ============================

st.subheader("📊 Resultado da Predição")

if st.sidebar.button("🔍 Calcular Risco"):
    try:
        pred = model.predict(df)[0]

        if pred == 0:
            st.success("💰 **ADIMPLENTE (0)** — baixa probabilidade de inadimplência.")
        elif pred == 3:
            st.error("⚠️ **INADIMPLENTE (3)** — alta probabilidade.")
        else:
            st.warning(f"Resultado inesperado: {pred}")
    except Exception as e:
        st.error(f"❌ Erro ao realizar predição: {e}")

else:
    st.info("Clique no botão para calcular o risco.")

st.markdown("---")

# ============================
# 7) GRÁFICO DAS FEATURES
# ============================

st.subheader("📈 Valores Informados")

base_plot = {
    "Variável": ["Renda", "Idade", "Score", "Valor", "Meses atrasados"],
    "Valor": [renda, idade, score, valor, meses],
}

fig = px.bar(base_plot, x="Variável", y="Valor", title="Informações do Estudante", text="Valor")
st.plotly_chart(fig, use_container_width=True)
