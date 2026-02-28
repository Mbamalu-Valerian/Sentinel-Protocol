from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class FraudDetector:
    def __init__(self):
        # Initialize Random Forest with 'balanced' class weights
        # This is CRITICAL for fraud detection because fraud is rare.
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',  # <--- The secret sauce
            n_jobs=-1  # Use all CPU cores
        )

    def train(self, X_train, y_train):
        print("Training Random Forest Model (this takes the most time)...")
        self.model.fit(X_train, y_train)
        print("Training Complete!")

    def evaluate(self, X_test, y_test):
        print("\n--- Model Evaluation Report ---")
        predictions = self.model.predict(X_test)

        # This prints Precision, Recall, and F1-Score
        print(classification_report(y_test, predictions))

        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))

    def save_model(self, path="models/trained_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")