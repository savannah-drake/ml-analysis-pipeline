import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def run_regression(data_path="data/regression_data.csv"):
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Regression Model: Actual vs Predicted Values")
    os.makedirs("outputs/visuals", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)
    plt.savefig("outputs/visuals/regression_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    results = pd.DataFrame({
        "actual": y_test,
        "predicted": predictions
    })

    results.to_csv("outputs/regression_output.csv", index=False)

    return mse, r2, results