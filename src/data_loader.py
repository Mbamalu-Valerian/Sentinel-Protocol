import pandas as pd
import os


class DataLoader:
    def __init__(self, raw_data_path):
        """
        raw_data_path: Path to the folder containing 'elliptic_txs_classes.csv' and 'elliptic_txs_features.csv'
        """
        self.raw_data_path = raw_data_path

    def load_data(self):
        print("Loading classes...")
        classes_file = os.path.join(self.raw_data_path, 'elliptic_txs_classes.csv')
        df_classes = pd.read_csv(classes_file)

        print("Loading features (this might take a minute)...")
        features_file = os.path.join(self.raw_data_path, 'elliptic_txs_features.csv')
        # The features file has no headers, just raw numbers. We rename the first 2 columns.
        df_features = pd.read_csv(features_file, header=None)

        # Rename columns: Col 0 is txId, Col 1 is time_step
        col_names = ['txId', 'time_step'] + [f'trans_feat_{i}' for i in range(93)] + [f'agg_feat_{i}' for i in
                                                                                      range(72)]
        df_features.columns = col_names

        print("Merging data...")
        # Merge the Class labels with the Features
        data = pd.merge(df_features, df_classes, on='txId', how='left')

        # Filter out 'unknown' classes (we can only train on Licit (2) vs Illicit (1))
        # Map: 1 (Illicit) -> 1, 2 (Licit) -> 0
        data['class'] = data['class'].map({'1': 1, '2': 0, 'unknown': -1})

        # Drop unknowns for training
        clean_data = data[data['class'] != -1].copy()

        print(f"Data Loaded! Shape: {clean_data.shape}")
        return clean_data


# Test the code if run directly
if __name__ == "__main__":
    # Adjust this path to where your zip file contents are
    path = "data/raw/"
    loader = DataLoader(path)
    df = loader.load_data()
    print(df.head())