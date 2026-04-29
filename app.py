import streamlit as st
from src.clustering import run_clustering
from src.classification import run_classification
from src.regression import run_regression
from PIL import Image

st.set_page_config(page_title="ML Analysis", layout="centered")

st.title("Machine Learning Analysis Dashboard")

st.write(
    "Explore how regression, classification, and clustering provide different insights from the same data."
)

task = st.selectbox(
    "Choose a model to run:",
    ["Clustering", "Classification", "Regression"]
)

if task == "Clustering":
    st.subheader("Clustering Results")
    run_clustering()
    img = Image.open("outputs/visuals/clustering_results.png")
    st.image(img, caption="Cluster Visualization")

elif task == "Classification":
    st.subheader("Classification Results")
    accuracy, _ = run_classification()
    st.write(f"Accuracy: {accuracy:.4f}")
    img = Image.open("outputs/visuals/classification_confusion_matrix.png")
    st.image(img, caption="Confusion Matrix")

elif task == "Regression":
    st.subheader("Regression Results")
    mse, r2, _ = run_regression()
    st.write(f"MSE: {mse:.4f}")
    st.write(f"R²: {r2:.4f}")
    img = Image.open("outputs/visuals/regression_actual_vs_predicted.png")
    st.image(img, caption="Actual vs Predicted")

st.markdown("---")
st.caption("Built by Savannah Drake")