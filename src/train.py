"""
Training script for the Coupon Recommendation System.

Loads the raw dataset, cleans it, builds a preprocessing + RandomForest
pipeline, tunes hyperparameters with GridSearchCV, evaluates on a held-out
test set, and saves the final trained pipeline to disk for the Streamlit app.

Usage:
    python src/train.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = "data/DS_DATA.csv"
MODEL_OUTPUT_PATH = "coupon_model.pkl"


def load_and_clean_data(path: str):
    """Load the raw CSV and apply basic cleaning."""
    df = pd.read_csv(path)

    # Drop column with heavy missingness
    df_clean = df.drop(columns=["car"])

    # Fill missing values in behavioral frequency columns with the mode
    behavioral_cols = [
        "Bar", "CoffeeHouse", "CarryAway",
        "RestaurantLessThan20", "Restaurant20To50",
    ]
    for col in behavioral_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    X = df_clean.drop(columns=["Accept(Y/N?)"])
    y = df_clean["Accept(Y/N?)"]
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build the preprocessing + RandomForest pipeline."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    return pipeline


def main():
    print("Loading and cleaning data...")
    X, y = load_and_clean_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(X)

    param_grid = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [10, 15, 20, None],
        "classifier__min_samples_split": [2, 5],
    }

    print("Running GridSearchCV (5-fold cross-validation)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print("\n--- SEARCH RESULTS ---")
    print("Best Hyperparameters:", grid_search.best_params_)
    print(f"Best CV Accuracy: {grid_search.best_score_ * 100:.2f}%")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print(f"\nFinal Test Set Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Accepted (1)"]))

    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"\nSaved trained pipeline to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
