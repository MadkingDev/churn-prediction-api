import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # adjust if needed
MODEL_OUTPUT_PATH = "churn_model.pkl"

EXPERIMENT_NAME = "churn_prediction_experiment"

# ----------------------------
# Load Data
# ----------------------------
def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


# ----------------------------
# Build Pipeline
# ----------------------------
def build_pipeline(X):

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                ),
            ),
        ]
    )

    return pipeline


# ----------------------------
# Training Function
# ----------------------------
def train():

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():

        df = load_data(DATA_PATH)

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        pipeline = build_pipeline(X)

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Accuracy:", accuracy)
        print(classification_report(y_test, y_pred))

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        # Save locally for FastAPI
        joblib.dump(pipeline, os.path.join(os.path.dirname(__file__), MODEL_OUTPUT_PATH))

        print("Model saved successfully!")


if __name__ == "__main__":
    train()
