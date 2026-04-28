import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import joblib.numpy_pickle as joblib_numpy_pickle
import shap
import os

# --- Compatibility shim for legacy sklearn pickles ---
class _CompatRemainderColsList(list):
    pass

_original_joblib_unpickler_find_class = joblib_numpy_pickle.NumpyUnpickler.find_class

def _compat_find_class(self, module, name):
    if module == 'sklearn.compose._column_transformer' and name == '_RemainderColsList':
        return _CompatRemainderColsList
    return _original_joblib_unpickler_find_class(self, module, name)

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Assets (FIXED PATHS) ---
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    joblib_numpy_pickle.NumpyUnpickler.find_class = _compat_find_class
    try:
        model = joblib.load(os.path.join(BASE_DIR, 'models/random_forest_model.pkl'))
        preprocessor = joblib.load(os.path.join(BASE_DIR, 'models/preprocessor.pkl'))
    finally:
        joblib_numpy_pickle.NumpyUnpickler.find_class = _original_joblib_unpickler_find_class

    X_test = joblib.load(os.path.join(BASE_DIR, 'models/X_test.pkl'))
    combined_df = pd.read_csv(os.path.join(BASE_DIR, 'data/combined_df.csv'))

    explainer = shap.TreeExplainer(model)
    return model, preprocessor, X_test, combined_df, explainer


# --- 3. Safe Load ---
try:
    model, preprocessor, X_test, combined_df, explainer = load_assets()
except Exception as e:
    st.exception(e)
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("📌 Navigation")
    st.markdown("""
    **Project Overview**  
    Predicts student final grades (**G3**) using a Random Forest model.
    
    **Instructions**  
    1. Enter student details  
    2. Click **Analyze Performance**
    """)
    st.divider()
    st.caption("Kaggle Alcohol Study Analysis")

# --- Header ---
st.title("🎓 Student Academic Performance Dashboard")
st.divider()

# --- Layout ---
col1, col2 = st.columns([0.6, 0.4], gap="large")

with col1:
    st.header("Student Profile")

    with st.expander("Demographic", expanded=True):
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Area", ["U", "R"])
        famsize = st.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = st.selectbox("Parental Status", ["T", "A"])

    with st.expander("Academic", expanded=True):
        failures = st.number_input("Past Failures", 0, 4, 0)
        absences = st.number_input("Absences", 0, 100, 5)
        studytime = st.select_slider("Study Time", options=[1, 2, 3, 4])
        goout = st.slider("Socializing", 1, 5, 3)

    with st.expander("Lifestyle"):
        Dalc = st.slider("Workday Alcohol", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol", 1, 5, 1)
        higher = st.checkbox("Higher Ed Goal", value=True)

    analyze = st.button("Analyze Performance", use_container_width=True)

with col2:
    st.header("Prediction")

    if analyze:
        # Avoid division by zero cleanly
        study_efficiency = studytime / absences if absences > 0 else studytime

        raw_data = pd.DataFrame([{
            'school': school, 'sex': sex, 'age': age, 'address': address,
            'famsize': famsize, 'Pstatus': Pstatus,
            'Medu': 2, 'Fedu': 2,
            'Mjob': 'other', 'Fjob': 'other',
            'reason': 'course', 'guardian': 'mother',
            'traveltime': 1, 'studytime': studytime,
            'failures': failures,
            'schoolsup': 'no', 'famsup': 'no',
            'paid': 'no', 'activities': 'no',
            'nursery': 'no',
            'higher': 'yes' if higher else 'no',
            'internet': 'yes',
            'romantic': 'no',
            'famrel': 3, 'freetime': 3,
            'goout': goout,
            'Dalc': Dalc, 'Walc': Walc,
            'health': 3,
            'absences': absences,
            'total_alcohol_consumption': Dalc + Walc,
            'study_efficiency': study_efficiency,
            'G1': 0, 'G2': 0, 'G3': 0  # required by the saved preprocessor input schema
        }])

        processed_df = pd.DataFrame(
            preprocessor.transform(raw_data),
            columns=X_test.columns
        )

        prediction = model.predict(processed_df)[0]

        st.metric("Predicted Grade", f"{prediction:.2f} / 20")

        if prediction >= 10:
            st.success("PASS")
        else:
            st.error("AT RISK")

# --- Global Insights ---
st.divider()
st.header("Global Insights")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    sns.boxplot(x='Walc', y='G3', data=combined_df, ax=ax)
    ax.set_title("Weekend Alcohol vs Final Grade")
    st.pyplot(fig)

with c2:
    importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values().tail(8)

    fig2, ax2 = plt.subplots()
    importance.plot(kind='barh', ax=ax2)
    ax2.set_title("Top Feature Importances")
    st.pyplot(fig2)