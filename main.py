import argparse
from src.clustering import run_clustering
from src.classification import run_classification
from src.regression import run_regression


def main():
    parser = argparse.ArgumentParser(description="Run ML project tasks.")
    parser.add_argument(
        "--task",
        choices=["clustering", "classification", "regression", "all"],
        required=True,
        help="Choose which task to run."
    )

    args = parser.parse_args()

    if args.task == "clustering":
        output = run_clustering()
        print("Clustering complete.")
        print(output.head())

    elif args.task == "classification":
        accuracy, output = run_classification()
        print("Classification complete.")
        print(f"Accuracy: {accuracy:.4f}")
        print(output.head())

    elif args.task == "regression":
        mse, r2, output = run_regression()
        print("Regression complete.")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        print(output.head())

    elif args.task == "all":
        clustering_output = run_clustering()
        accuracy, classification_output = run_classification()
        mse, r2, regression_output = run_regression()

        print("All tasks complete.")
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Regression MSE: {mse:.4f}")
        print(f"Regression R²: {r2:.4f}")


if __name__ == "__main__":
    main()