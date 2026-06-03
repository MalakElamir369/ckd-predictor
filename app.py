
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="CKD Predictor",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_pipeline():
    return joblib.load("ckd_random_forest_pipeline.pkl")

pipeline = load_pipeline()
scaler = pipeline["scaler"]
selector = pipeline["selector"]
model = pipeline["model"]
feature_names = pipeline["feature_names"]

st.title("🏥 CKD Prediction App")
st.write(
    "Enter the patient's clinical values below to predict whether they have Chronic Kidney Disease."
)

st.subheader("Numerical Features")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 100, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.number_input("Specific Gravity", 1.000, 1.030, 1.015, format="%.3f")
    al = st.number_input("Albumin", 0, 5, 1)

with col2:
    su = st.number_input("Sugar", 0, 5, 0)
    bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
    bu = st.number_input("Blood Urea", 1, 400, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2, format="%.1f")

with col3:
    sod = st.number_input("Sodium", 100, 200, 138)
    pot = st.number_input("Potassium", 2, 10, 4)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5, format="%.1f")
    pcv = st.number_input("Packed Cell Volume", 10, 60, 40)

col4, col5 = st.columns(2)

with col4:
    wc = st.number_input(
        "White Blood Cell Count",
        min_value=2000,
        max_value=26000,
        value=8000
    )

with col5:
    rc = st.number_input(
        "Red Blood Cell Count",
        min_value=2.0,
        max_value=8.0,
        value=5.0,
        format="%.1f"
    )

st.subheader("Categorical Features")

col6, col7, col8 = st.columns(3)

with col6:
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cells", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent", "present"])
    ba = st.selectbox("Bacteria", ["notpresent", "present"])

with col7:
    htn = st.selectbox("Hypertension", ["no", "yes"])
    dm = st.selectbox("Diabetes Mellitus", ["no", "yes"])
    cad = st.selectbox("Coronary Artery Disease", ["no", "yes"])

with col8:
    appet = st.selectbox("Appetite", ["good", "poor"])
    pe = st.selectbox("Pedal Edema", ["no", "yes"])
    ane = st.selectbox("Anemia", ["no", "yes"])

binary_mappings = {
    "rbc": {"normal": 1, "abnormal": 0},
    "pc": {"normal": 1, "abnormal": 0},
    "pcc": {"present": 1, "notpresent": 0},
    "ba": {"present": 1, "notpresent": 0},
    "htn": {"yes": 1, "no": 0},
    "dm": {"yes": 1, "no": 0},
    "cad": {"yes": 1, "no": 0},
    "pe": {"yes": 1, "no": 0},
    "ane": {"yes": 1, "no": 0},
    "appet": {"good": 1, "poor": 0},
}

def encode(column, value):
    return binary_mappings[column][value]

if st.button("🔍 Predict"):

    raw = {
        "age": age,
        "bp": bp,
        "sg": sg,
        "al": al,
        "su": su,
        "rbc": encode("rbc", rbc),
        "pc": encode("pc", pc),
        "pcc": encode("pcc", pcc),
        "ba": encode("ba", ba),
        "bgr": bgr,
        "bu": bu,
        "sc": sc,
        "sod": sod,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "wc": wc,
        "rc": rc,
        "htn": encode("htn", htn),
        "dm": encode("dm", dm),
        "cad": encode("cad", cad),
        "appet": encode("appet", appet),
        "pe": encode("pe", pe),
        "ane": encode("ane", ane),
    }

    raw["age_sc_interaction"] = age * sc
    raw["hemo_bp_ratio"] = hemo / (bp + 1)

    input_df = pd.DataFrame([raw], columns=feature_names)

    scaled = scaler.transform(input_df)
    selected = selector.transform(scaled)

    result = model.predict(selected)[0]
    proba = model.predict_proba(selected)[0]

    st.divider()

    if result == 1:
        st.error(f"⚠️ CKD Detected — Confidence: {proba[1] * 100:.1f}%")
        st.warning(
            "This patient is likely to have Chronic Kidney Disease. Please consult a specialist."
        )
    else:
        st.success(f"✅ No CKD Detected — Confidence: {proba[0] * 100:.1f}%")
        st.success(
            "This patient is unlikely to have Chronic Kidney Disease."
        )

    st.subheader("Prediction Probabilities")

    prob_df = pd.DataFrame(
        {
            "Condition": ["Not CKD", "CKD"],
            "Probability": [
                f"{proba[0] * 100:.1f}%",
                f"{proba[1] * 100:.1f}%"
            ]
        }
    )

    st.table(prob_df)

st.divider()
st.caption(
    "Best Model Selected After Comparison: Random Forest | Dataset: UCI Chronic Kidney Disease"
)
