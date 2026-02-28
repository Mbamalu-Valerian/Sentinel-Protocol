import os

from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from features import FeatureEngineer
from model import FraudDetector


def main():
    # 1. Load Data
    print("STEP 1: Loading Data...")
    loader = DataLoader("data/raw/")
    df = loader.load_data()

    # 2. Engineer Features
    print("STEP 2: Engineering Features...")
    engineer = FeatureEngineer("data/raw/")
    df = engineer.add_graph_features(df)

    # 3. Prepare for Training
    print("STEP 3: Preparing Training Sets...")
    # Drop columns we can't use for math (like ID)
    # We keep the new features: in_degree, out_degree, entity_complexity
    X = df.drop(columns=['txId', 'class', 'time_step'])
    y = df['class']

    # Split: 70% Training, 30% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Train Model
    print("STEP 4: Training...")
    detector = FraudDetector()
    detector.train(X_train, y_train)

    # 5. Evaluate
    print("STEP 5: Evaluation...")
    detector.evaluate(X_test, y_test)

    # 6. Save
    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    detector.save_model("models/trained_model.pkl")
    print("\nSUCCESS! System is built and model is saved.")


if __name__ == "__main__":
    main()