import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FraudModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None

    def load_and_split_data(self):
        logging.info("Loading processed data...")
        df = pd.read_csv(self.data_path)

        # Define Features (X) and Target (y)
        # We drop 'txId', 'class', and 'is_fraud' from features.
        # 'class' is the string label ("1", "2"), 'is_fraud' is the binary target (1, 0)
        X = df.drop(columns=['txId', 'class', 'is_fraud'])
        y = df['is_fraud']

        # Save feature names for later use in the dashboard
        self.feature_names = X.columns.tolist()

        logging.info(f"Features selected: {len(self.feature_names)}")

        # Split: 70% for Training, 30% for Testing
        # random_state=42 ensures we get the same split every time (good for defense consistency)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        logging.info(f"Data split. Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        logging.info("Initializing Random Forest Classifier...")

        # class_weight='balanced' is the KEY to handling your imbalance problem.
        # It tells the model: "Pay 10x more attention to fraud because it's rare."
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Uses all your CPU cores to train faster
        )

        logging.info("Training started (this might take 1-2 minutes)...")
        self.model.fit(X_train, y_train)
        logging.info("Training complete.")

    def evaluate(self, X_test, y_test):
        logging.info("Evaluating model performance...")
        y_pred = self.model.predict(X_test)

        # Print the detailed report (Precision, Recall, F1-Score)
        report = classification_report(y_test, y_pred)
        print("\n--- Model Performance Report ---")
        print(report)
        print("--------------------------------")

        return report

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'random_forest_model.pkl')
        columns_path = os.path.join(output_dir, 'model_columns.pkl')

        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_names, columns_path)
        logging.info(f"Model saved to {model_path}")
        logging.info(f"Column names saved to {columns_path}")


# --- Run Script ---
if __name__ == "__main__":
    base_path = os.getcwd()
    # Input file (from previous step)
    data_file = os.path.join(base_path, 'data', 'processed', 'train_ready_data.csv')
    # Output folder
    models_dir = os.path.join(base_path, 'models')

    trainer = FraudModelTrainer(data_file)

    # 1. Load & Split
    X_train, X_test, y_train, y_test = trainer.load_and_split_data()

    # 2. Train
    trainer.train(X_train, y_train)

    # 3. Evaluate
    trainer.evaluate(X_test, y_test)

    # 4. Save
    trainer.save_model(models_dir)

    print("\nSUCCESS! Model Trained and Saved.")