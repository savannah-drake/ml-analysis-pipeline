import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


def run_classification(data_path="data/classification_data.csv"):
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot()
    plt.title("Model Performance: Confusion Matrix")
    os.makedirs("outputs/visuals", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    plt.savefig("outputs/visuals/classification_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    results = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })

    results.to_csv("outputs/classification_output.csv", index=False)

    return accuracy, results