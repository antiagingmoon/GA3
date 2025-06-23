import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
from backend import load_patient_data, predict_cancer_type

# 페이지 설정
st.set_page_config(page_title="환자 유전자 발현 분석기", layout="wide")

# 맞춤 CSS - 파란 테마 적용
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
        }
        .main {
            background-color: #ffffff;
        }
        h1, h2, h3 {
            color: #0b5394;
        }
        .stButton>button {
            color: white;
            background-color: #0b5394;
            border: none;
        }
        .stCheckbox>label>div {
            color: #0b5394;
        }
        .stSelectbox>div>div>div {
            color: #0b5394;
        }
    </style>
""", unsafe_allow_html=True)

# 제목 및 소개
st.title("🧬 환자 유전자 발현 분석기")
st.write("""
유전자 발현 데이터와 메타데이터를 업로드하세요.  
앱은 환자의 건강 상태를 메타데이터를 기반으로 판단하고,  
암 환자의 경우 암 종류를 예측합니다.  
Heatmap, Volcano Plot, PCA를 통해 데이터를 시각화할 수도 있습니다.
""")

# 파일 업로드
expression_file = st.file_uploader("📄 유전자 발현 파일 업로드 (.xlsx)", type=["xlsx"])
metadata_file = st.file_uploader("📄 환자 메타데이터 파일 업로드 (.xlsx)", type=["xlsx"])

if expression_file and metadata_file:
    expr_df, meta_df = load_patient_data(expression_file, metadata_file)

    st.subheader("📊 업로드된 유전자 발현 데이터")
    st.dataframe(expr_df)

    st.subheader("📋 업로드된 환자 메타데이터")
    st.dataframe(meta_df)

    st.subheader("🧠 건강 평가 및 예측된 암 종류")

    condition_list = []
    predicted_types = []

    for _, row in meta_df.iterrows():
        patient_id = row["PatientID"]
        condition = row["Condition"].strip().capitalize()

        if condition == "Cancer":
            predicted_type = predict_cancer_type(expr_df, patient_id)
            condition_list.append("질환 있음")
            predicted_types.append(predicted_type)
        else:
            condition_list.append("건강함")
            predicted_types.append("해당 없음")

    results_df = meta_df.copy()
    results_df["건강 평가"] = condition_list
    results_df["예측된 암 종류"] = predicted_types

    st.dataframe(results_df)

    st.subheader("🔍 개별 환자 리포트")
    selected_patient = st.selectbox("환자 ID를 선택하세요", meta_df["PatientID"])

    if selected_patient:
        patient_info = results_df[results_df["PatientID"] == selected_patient].iloc[0]
        st.markdown(f"### 🧾 `{selected_patient}`에 대한 진단 결과")
        st.write(f"- 건강 상태: **{patient_info['건강 평가']}**")
        if patient_info["건강 평가"] == "질환 있음":
            st.write(f"- 예측된 암 종류: **{patient_info['예측된 암 종류']}**")
        else:
            st.write("- 환자는 건강합니다. 암의 징후는 없습니다.")

    # --- 시각화 ---
    st.header("📊 유전자 발현 차이 시각화")

    if st.checkbox("🔬 히트맵 보기"):
        st.subheader("🌡️ 유전자 발현 히트맵")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(expr_df, cmap="Blues", ax=ax)
        st.pyplot(fig)

    if st.checkbox("🌋 볼케이노 플롯 보기"):
        st.subheader("🌋 Volcano Plot (발현 차이 시각화)")

        healthy_ids = meta_df[meta_df["Condition"].str.lower() == "healthy"]["PatientID"]
        cancer_ids = meta_df[meta_df["Condition"].str.lower() == "cancer"]["PatientID"]

        log2_fc = []
        p_values = []

        for gene in expr_df.index:
            group1 = expr_df.loc[gene, healthy_ids]
            group2 = expr_df.loc[gene, cancer_ids]
            fc = np.log2((group2.mean() + 1e-6) / (group1.mean() + 1e-6))
            p = ttest_ind(group2, group1, equal_var=False).pvalue
            log2_fc.append(fc)
            p_values.append(p)

        volcano_df = pd.DataFrame({
            "Gene": expr_df.index,
            "log2FC": log2_fc,
            "-log10(p-value)": -np.log10(p_values)
        })

        fig, ax = plt.subplots()
        sns.scatterplot(data=volcano_df, x="log2FC", y="-log10(p-value)", ax=ax,
                        hue=volcano_df["-log10(p-value)"] > 1.3, palette="coolwarm")
        ax.axhline(y=1.3, color='red', linestyle='--')
        ax.axvline(x=1, color='blue', linestyle='--')
        ax.axvline(x=-1, color='blue', linestyle='--')
        st.pyplot(fig)

    if st.checkbox("📈 PCA 플롯 보기"):
        st.subheader("📈 PCA (주성분 분석) 플롯")

        X = expr_df.T
        y = meta_df.set_index("PatientID").loc[X.index]["Condition"]

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Condition"] = y.values

        fig, ax = plt.subplots()
        sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", s=100, palette="Blues")
        st.pyplot(fig)
