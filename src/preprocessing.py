import logging
import os

import networkx as nx

from data_loader import EllipticDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineer:
    def __init__(self, df_data, df_edges):
        """
        Initialize with the loaded dataframes.
        """
        self.df = df_data
        self.edges = df_edges

    def add_network_features(self):
        """
        Calculates graph metrics (Degree Centrality) for each transaction.
        This fulfills specific objective #2 of your project.
        """
        logging.info("Building Network Graph to extract features...")

        # Create a graph from the edges list
        # Directed graph because money flows FROM -> TO
        G = nx.from_pandas_edgelist(self.edges, source='txId1', target='txId2', create_using=nx.DiGraph())

        logging.info(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        # Calculate Degree Centrality (How many people represent the inputs/outputs)
        # In-degree = Money received
        # Out-degree = Money sent
        logging.info("Calculating In-Degree and Out-Degree centrality (this may take a moment)...")
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # Map these features back to our main dataframe
        # We use 'map' to look up the txId in the degree dictionary
        self.df['in_degree'] = self.df['txId'].map(in_degree).fillna(0)
        self.df['out_degree'] = self.df['txId'].map(out_degree).fillna(0)

        # Calculate Clustering Coefficient (optional but powerful for defense)
        # Note: This is computationally heavy. For a demo, degrees are usually enough.
        # We will stick to degrees to keep it fast for your laptop.

        logging.info("Network features added.")
        return self.df

    def save_processed_data(self, output_path):
        """
        Saves the final clean dataset ready for training.
        """
        try:
            self.df.to_csv(output_path, index=False)
            logging.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving data: {e}")


# --- Robust Test Block ---
if __name__ == "__main__":
    # 1. Setup Paths
    base_path = os.getcwd()
    raw_path = os.path.join(base_path, 'data', 'raw')
    processed_path = os.path.join(base_path, 'data', 'processed')

    # Create processed folder if it doesn't exist
    os.makedirs(processed_path, exist_ok=True)
    output_file = os.path.join(processed_path, 'train_ready_data.csv')

    # 2. Load Data (Using the class we made in Step 1)
    loader = EllipticDataLoader(raw_path)
    df_train, df_edges = loader.load_data()

    # 3. Engineer Features
    engineer = FeatureEngineer(df_train, df_edges)
    df_final = engineer.add_network_features()

    # 4. Save
    engineer.save_processed_data(output_file)

    print("\nSUCCESS! Feature Engineering Complete.")
    print(f"New Features Added: 'in_degree', 'out_degree'")
    print(f"Final Data Shape: {df_final.shape}")
    print(f"Saved to: {output_file}")