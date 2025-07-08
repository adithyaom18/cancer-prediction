# app/main.py

import streamlit as st
import numpy as np
import pickle

# Load all models, scaler, and imputer
with open('model/logreg.pkl', 'rb') as f:
    logreg_model = pickle.load(f)
with open('model/rf.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('model/svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('model/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

# Load metrics
with open('model/metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)


# Load model accuracy scores
with open('model/scores.pkl', 'rb') as f:
    scores = pickle.load(f)


# Model selection map
model_options = {
    'Logistic Regression': logreg_model,
    'Random Forest': rf_model,
    'Support Vector Machine': svm_model
}

# Streamlit UI
st.title("🩺 Breast Cancer Prediction App")
# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=100)
    st.title("⚙️ Settings")

    selected_model_name = st.selectbox("Choose ML Model", list(model_options.keys()))
    
    st.markdown("---")
    st.markdown("📊 **Model Metrics (Test Data):**")
    
    name_map = {
        "logreg": "Logistic Regression",
        "rf": "Random Forest",
        "svm": "SVM"
    }

    for key, metric in metrics.items():
        name = name_map.get(key, key.upper())
        st.markdown(f"**{name}**")
        st.markdown(f"- Accuracy: `{metric['accuracy']*100:.2f}%`")
        st.markdown(f"- Precision: `{metric['precision']*100:.2f}%`")
        st.markdown(f"- Recall: `{metric['recall']*100:.2f}%`")
        st.markdown(f"- F1 Score: `{metric['f1_score']*100:.2f}%`")
        st.markdown("---")

    st.markdown("💡 This model uses features extracted from digitized breast tumor biopsies to predict cancer type.")
    st.markdown("🔗 [GitHub Repo](https://github.com/YOUR_USERNAME/YOUR_REPO)")
    st.markdown("📫 Contact: [you@example.com](mailto:you@example.com)")



selected_model = model_options[selected_model_name]

st.markdown("""
This app predicts whether a tumor is **benign or malignant** using different machine learning models.
""")

st.subheader("Enter Tumor Features:")

# Define 30 input fields (based on dataset)
features = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

feature_ranges = {
    "radius_mean": (5.0, 30.0),
    "texture_mean": (5.0, 40.0),
    "perimeter_mean": (40.0, 200.0),
    "area_mean": (100.0, 2500.0),
    "smoothness_mean": (0.05, 0.2),
    "compactness_mean": (0.01, 1.0),
    "concavity_mean": (0.0, 1.0),
    "concave points_mean": (0.0, 1.0),
    "symmetry_mean": (0.1, 0.5),
    "fractal_dimension_mean": (0.04, 0.2),

    "radius_se": (0.0, 3.0),
    "texture_se": (0.0, 5.0),
    "perimeter_se": (0.0, 15.0),
    "area_se": (0.0, 100.0),
    "smoothness_se": (0.0, 0.05),
    "compactness_se": (0.0, 0.2),
    "concavity_se": (0.0, 0.3),
    "concave points_se": (0.0, 0.2),
    "symmetry_se": (0.0, 0.1),
    "fractal_dimension_se": (0.0, 0.05),

    "radius_worst": (10.0, 40.0),
    "texture_worst": (10.0, 50.0),
    "perimeter_worst": (70.0, 250.0),
    "area_worst": (200.0, 4000.0),
    "smoothness_worst": (0.1, 0.3),
    "compactness_worst": (0.1, 1.2),
    "concavity_worst": (0.0, 1.5),
    "concave points_worst": (0.0, 1.5),
    "symmetry_worst": (0.2, 0.9),
    "fractal_dimension_worst": (0.05, 0.3),
}

# Helper to create synced input + slider
def synced_input(feature, min_val, max_val):
    key_input = feature + "_input"
    key_slider = feature + "_slider"

    # Initialize default value
    if key_input not in st.session_state:
        st.session_state[key_input] = float((min_val + max_val) / 2)
    if key_slider not in st.session_state:
        st.session_state[key_slider] = st.session_state[key_input]

    # Sync slider with number input
    def update_input():
        st.session_state[key_input] = st.session_state[key_slider]

    def update_slider():
        st.session_state[key_slider] = st.session_state[key_input]

    col1, col2 = st.columns([2, 3])

    with col1:
        st.number_input(
            f"{feature}", min_value=min_val, max_value=max_val,
            step=0.01, key=key_input, on_change=update_slider
        )

    with col2:
        st.slider(
            f"{feature}", min_value=min_val, max_value=max_val,
            step=0.01, key=key_slider, on_change=update_input
        )

    return st.session_state[key_slider]



# Grouped features
mean_features = [f for f in features if "_mean" in f]
se_features = [f for f in features if "_se" in f]
worst_features = [f for f in features if "_worst" in f]

user_input = []

st.subheader("📊 Mean Features")
for feature in mean_features:
    min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
    val = synced_input(feature, float(min_val), float(max_val))
    user_input.append(val)

st.subheader("📉 Standard Error Features")
for feature in se_features:
    min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
    val = synced_input(feature, float(min_val), float(max_val))
    user_input.append(val)

st.subheader("⚠️ Worst Case Features")
for feature in worst_features:
    min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))
    val = synced_input(feature, float(min_val), float(max_val))
    user_input.append(val)



if st.button("Predict"):
    input_array = np.array(user_input).reshape(1, -1)
    input_imputed = imputer.transform(input_array)
    input_scaled = scaler.transform(input_imputed)
    prediction = selected_model.predict(input_scaled)[0]
    probability = selected_model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🔴 The model predicts this tumor is **Malignant** with {probability*100:.2f}% confidence.")
    else:
        st.success(f"🟢 The model predicts this tumor is **Benign** with {probability*100:.2f}% confidence.")
