import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import missingno as msno
import nannyml as nml
from nannyml import MissingValuesCalculator, UnseenValuesCalculator, AlertCountRanker
from datetime import timedelta
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="Dashboard de Fraudes", layout="wide")
st.title("🔎 Dashboard de Fraudes")

# Seletor lateral
aba = st.sidebar.radio("📁 Selecione uma seção:", ["📊 Gráficos", "📡 Monitoramento"])

# ==========================================================================
# 📊 GRÁFICOS
# ==========================================================================
if aba == "📊 Gráficos":
    st.header("📊 Análise de Detecção de Fraudes")

    df = pd.read_csv("../data/processed/result_with_predictions.csv")

    threshold = 61
    df["predicted_proba"] = pd.to_numeric(df["predicted_proba"], errors="coerce")
    df["true_fraude"] = pd.to_numeric(df["true_fraude"], errors="coerce")
    df["valor_compra"] = pd.to_numeric(df["valor_compra"], errors="coerce")

    df["blocked"] = df["predicted_proba"] * 100 < threshold
    df["predicted_fraude_threshold"] = (df["predicted_proba"] * 100 >= threshold).astype(int)

    st.subheader("📈 Histograma dos Scores de Fraude")
    fig = px.histogram(df, x="predicted_proba", nbins=30)
    fig.add_vline(x=threshold / 100, line_dash="dash", line_color="red", annotation_text=f"Threshold = {threshold/100:.2f}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Métricas de Classificação")
    report = classification_report(df["true_fraude"], df["predicted_fraude_threshold"], output_dict=True)
    metricas = {
        "Accuracy": report["accuracy"],
        "Precision (fraude)": report["1"]["precision"],
        "Recall (fraude)": report["1"]["recall"],
        "F1-Score (fraude)": report["1"]["f1-score"],
    }
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metricas['Accuracy']:.2%}")
    col2.metric("Precision", f"{metricas['Precision (fraude)']:.2%}")
    col3.metric("Recall", f"{metricas['Recall (fraude)']:.2%}")
    col4.metric("F1-Score", f"{metricas['F1-Score (fraude)']:.2%}")

    st.subheader("🧮 Matrizes de Confusão")
    cm = confusion_matrix(df["true_fraude"], df["predicted_fraude_threshold"])
    cm_prop = cm / cm.sum(axis=1, keepdims=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=["Predito 0", "Predito 1"], y=["Real 0", "Real 1"],
            text=cm, texttemplate="%{text}", colorscale="Blues"
        ))
        fig_cm.update_layout(title="Matriz de Confusão (Absoluta)")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        fig_cm_prop = go.Figure(data=go.Heatmap(
            z=np.round(cm_prop, 2), x=["Predito 0", "Predito 1"], y=["Real 0", "Real 1"],
            text=np.round(cm_prop * 100, 2), texttemplate="%{text}%", colorscale="Greens"
        ))
        fig_cm_prop.update_layout(title="Matriz de Confusão (Proporcional)")
        st.plotly_chart(fig_cm_prop, use_container_width=True)

    st.subheader("📉 Taxas Operacionais")
    TN, FP, FN, TP = cm.ravel()
    approval_rate = round((FN + TN) / (TN + FP + FN + TP), 2)
    fraud_rate = round(FN / (FN + TN), 2) if (FN + TN) > 0 else 0

    col1, col2 = st.columns(2)
    col1.metric("Taxa de Aprovação", f"{approval_rate:.2%}")
    col2.metric("Taxa de Fraude Aprovada", f"{fraud_rate:.2%}")

    st.subheader("💰 Análise Financeira")

    def profit_from_prediction(df, predicted_col, target_col, amount_col):
        fraud_losses = df[(df[predicted_col] == 0) & (df[target_col] == 1)][amount_col].sum()
        revenues = df[(df[predicted_col] == 0) & (df[target_col] == 0)][amount_col].sum() * 0.1
        profit = revenues - fraud_losses
        return fraud_losses, revenues, profit

    fraud_losses, revenues, profit = profit_from_prediction(df, "predicted_fraude_threshold", "true_fraude", "valor_compra")
    total = fraud_losses + revenues
    profit_ratio = profit / total if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Perdas com Fraudes", f"R$ {fraud_losses:,.2f}")
    col2.metric("Receitas", f"R$ {revenues:,.2f}")
    col3.metric("Lucro Líquido", f"R$ {profit:,.2f}")
    col4.metric("Razão de Lucro", f"{profit_ratio:.2%}")

# ==========================================================================
# 📡 MONITORAMENTO
# ==========================================================================
elif aba == "📡 Monitoramento":
    st.header("📡 Monitoramento: Comparação entre Treino e Inferência")

    df_train = pd.read_csv("../data/processed/train.csv")
    df_test = pd.read_csv("../data/processed/test.csv")

    # Limpeza
    colunas_para_remover = ["fraude", "score_fraude_modelo", "produto", "categoria_produto", "data_compra"]
    df_train = df_train.drop(columns=[col for col in colunas_para_remover if col in df_train.columns])
    df_test = df_test.drop(columns=[col for col in colunas_para_remover if col in df_test.columns])

    st.subheader("❓ Quantidade de Nulos")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Base de Treino**")
        fig1 = plt.figure(figsize=(6, 3))
        msno.bar(df_train)
        st.pyplot(fig1)

    with col2:
        st.write("**Base de Teste**")
        fig2 = plt.figure(figsize=(6, 3))
        msno.bar(df_test)
        st.pyplot(fig2)

    st.subheader("📈 Média de Variáveis Contínuas")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    mean_train = df_train[numeric_cols].mean()
    mean_test = df_test[numeric_cols].mean()
    means_df = pd.DataFrame({'Treino': mean_train, 'Teste': mean_test})
    st.line_chart(means_df)

    st.subheader("📊 Proporção de Variáveis Categóricas")
    categorical_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        st.write(f"**{col}**")
        p1 = df_train[col].value_counts(normalize=True)
        p2 = df_test[col].value_counts(normalize=True)
        prop_df = pd.DataFrame({"Treino": p1, "Teste": p2}).fillna(0)
        st.bar_chart(prop_df)

    # ========================
    # NANNYML - CBPE + Qualidade
    # ========================
    st.subheader("📡 Estimativa de Métricas com NannyML")

    df_train = pd.read_csv("../data/processed/result_with_predictions_train.csv")
    df_inference = pd.read_csv("../data/processed/result_with_predictions.csv")
    
    colunas_para_remover = ["fraude", "score_fraude_modelo", "produto", "categoria_produto", "data_compra"]
    df_train = df_train.drop(columns=[col for col in colunas_para_remover if col in df_train.columns])
    df_inference = df_inference.drop(columns=[col for col in colunas_para_remover if col in df_inference.columns])

    def simular_datas_repetidas(df, start_date="2024-01-01", n_dias=30):
        total = len(df)
        datas = pd.date_range(start=start_date, periods=n_dias, freq="D")
        datas_repetidas = np.resize(datas, total)
        df["prediction_date"] = np.sort(datas_repetidas)
        return df

    df_train = simular_datas_repetidas(df_train, "2023-12-01", 31)
    df_inference = simular_datas_repetidas(df_inference, "2024-01-01", 30)

    estimator = nml.CBPE(
        y_pred='predicted_fraude',
        y_pred_proba='predicted_proba',
        y_true='true_fraude',
        timestamp_column_name='prediction_date',
        problem_type='classification_binary',
        chunk_period='D',
        metrics=['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    )

    estimator.fit(df_train)
    estimated_performance = estimator.estimate(df_inference)

    fig_cbpe = estimated_performance.plot()
    st.plotly_chart(fig_cbpe, use_container_width=True)

    st.subheader("🔍 Qualidade de Dados com NannyML")

    exclude_cols = ['true_fraude', 'predicted_proba', 'predicted_fraude', 'valor_compra', 'prediction_date']
    features = [col for col in df_train.columns if col not in exclude_cols]

    # Valores Nulos
    missing_calc = MissingValuesCalculator(column_names=features, timestamp_column_name='prediction_date', chunk_period='D')
    missing_calc.fit(df_train)
    missing_results = missing_calc.calculate(df_inference)
    missing_ranking = AlertCountRanker().rank(missing_results)

    st.subheader("❓ Valores Nulos")
    st.dataframe(missing_ranking.head(10))
    fig_missing = missing_results.filter(column_names=missing_ranking['column_name'].head(3).tolist()).plot()
    st.plotly_chart(fig_missing, use_container_width=True)

    # Valores Não Observados
    st.subheader("🌐 Valores Não Observados")
    categorical_features = [col for col in features if df_train[col].dtype == 'object']
    unseen_calc = UnseenValuesCalculator(column_names=categorical_features, timestamp_column_name='prediction_date', chunk_period='D')
    unseen_calc.fit(df_train)
    unseen_results = unseen_calc.calculate(df_inference)
    unseen_ranking = AlertCountRanker().rank(unseen_results)

    st.dataframe(unseen_ranking.head(10))
    fig_unseen = unseen_results.filter(column_names=unseen_ranking['column_name'].head(3).tolist()).plot()
    st.plotly_chart(fig_unseen, use_container_width=True)
