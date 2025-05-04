import streamlit as st
import random
import pandas as pd
from utils import (
    load_models,
    load_feature_lists,
    load_model_performances,
    prepare_input_data,
    make_predictions,
    load_expected_answers
)

# Sabit 95 soruluk liste
def get_static_questions():
    return [
        'Q2','Q4','Q8','Q9','Q13','Q14','Q16','Q18','Q19','Q20','Q21','Q25','Q26','Q28','Q29',
        'Q33','Q34','Q35','Q40','Q44','Q45','Q47','Q51','Q52','Q53','Q54','Q60','Q62','Q67',
        'Q71','Q77','Q81','Q82','Q86','Q89','Q93','Q95','Q96','Q105','Q108','Q115','Q116',
        'Q117','Q119','Q125','Q126','Q127','Q128','Q129','Q130','Q133','Q138','Q139','Q140',
        'Q144','Q151','Q158','Q159','Q163','Q166','Q174','Q179','Q184','Q185','Q187','Q192',
        'Q197','Q202','Q203','Q204','Q205','Q210','Q212','Q215','Q219','Q221','Q222','Q224',
        'Q226','Q227','Q229','Q230','Q231','Q232','Q233','Q234','Q235','Q236','Q239','Q241',
        'Q242','Q243','Q249','Q252','Q253'
    ]

# Sayfa yapısı
st.set_page_config(page_title="Nörogelişimsel Bozukluk Tahmin Sistemi", layout="wide")
st.title("Nörogelişimsel Bozukluk Tahmin Sistemi")

if "page" not in st.session_state:
    st.session_state.page = "form"

questions = get_static_questions()

# Soru metinlerini yükle
@st.cache_data
def load_question_texts_local():
    df = pd.read_csv(
        "SorularFull.csv",
        sep=';',
        encoding='windows-1254',
        engine='python',
        quoting=3,
        quotechar=None,
        escapechar='\\'
    )
    df.columns = df.columns.str.strip().str.replace('"', '').str.lower()
    return {f"Q{idx+1}": row["soru"] for idx, row in df.iterrows() if f"Q{idx+1}" in questions}

question_texts = load_question_texts_local()

if st.session_state.page == "form":
    st.subheader("Lütfen aşağıdaki 95 soruyu cevaplayın")

    if st.button("Rastgele Doldur"):
        for q in questions:
            st.session_state[q] = random.choice(["Evet", "Hayır"])

    with st.form("questionnaire"):
        answers = {}
        for q in questions:
            label = question_texts.get(q, q)
            answers[q] = st.radio(
                label,
                ["Evet", "Hayır"],
                key=q,
                index=0 if st.session_state.get(q) == "Evet" else 1
            )
        submit = st.form_submit_button("Tahmin Yap")

    if submit:
        st.session_state.answers = answers
        st.session_state.page = "results"
        st.experimental_rerun()

elif st.session_state.page == "results":
    st.subheader("Cevaplarınız analiz ediliyor...")
    with st.spinner("Lütfen bekleyin. Tahmin yapılıyor..."):
        models = load_models()
        feature_lists = load_feature_lists()
        performances = load_model_performances()
        expected_answers = load_expected_answers()
        input_data = prepare_input_data(st.session_state.answers, feature_lists)
        results = make_predictions(models, input_data, performances, feature_lists, st.session_state.answers, expected_answers)

    st.subheader("Tahmin Sonuçları")

    for label, detail in results.items():
        if detail["final_prediction"] == 1:
            st.markdown(f"---\n### 📈 **{label}** – Eksiklik/Gelişimsel Risk Var")
            emoji = "🔴"
        else:
            st.markdown(f"---\n### ✅ **{label}** – Gelişim Normale Yakın")
            emoji = "🟢"
    
        st.markdown(f"- **Toplam Model Sayısı:** {detail['total_models']}")
        st.markdown(f"- **Eksiklik Diyen Model Sayısı:** {detail['total_positive']}")
        st.markdown(f"- **Risk Yüzdesi:** {emoji} **{detail['risk_percentage']:.1f}%**")
        st.markdown(f"- **Toplam Soru Sayısı:** {detail['total_question_count']}")
        st.markdown(f"- **Yanlış Cevaplanan Soru Sayısı:** {detail['incorrect_count']}")

        if detail.get("incorrect_answers_detailed"):
            with st.expander("🧩 Beklenenden farklı cevaplanan sorular"):
                for item in detail["incorrect_answers_detailed"]:
                    soru_kodu = item["soru_kodu"]
                    soru_metni = question_texts.get(soru_kodu, soru_kodu)
                    st.markdown(f"- **{soru_metni}**  \n"
                                f"*Beklenen Cevap:* `{item['beklenen']}`   |   *Verilen Cevap:* `{item['verilen']}`")


    if st.button("⬅️ Başa Dön"):
        st.session_state.page = "form"
        st.experimental_rerun()
