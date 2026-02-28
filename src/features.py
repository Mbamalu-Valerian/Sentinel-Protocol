import pandas as pd
import networkx as nx


class FeatureEngineer:
    def __init__(self, data_path):
        # We need the edgelist to build the graph
        self.edges_file = f"{data_path}/elliptic_txs_edgelist.csv"

    def add_graph_features(self, df):
        print("Building Network Graph (this might take a moment)...")
        # Load the edges (Who sent money to Whom)
        edges = pd.read_csv(self.edges_file)

        # Create a Directed Graph using NetworkX
        # Directed because money flows One Way (Source -> Target)
        G = nx.from_pandas_edgelist(edges, source='txId1', target='txId2', create_using=nx.DiGraph())

        print("Calculating Degree Centrality...")
        # In-degree: How many people sent money TO this wallet?
        in_degree = dict(G.in_degree())
        # Out-degree: How many people this wallet sent money TO?
        out_degree = dict(G.out_degree())

        # Map these new numbers to our existing Data
        # We use .map() to look up the txId in the graph and get the count
        df['in_degree'] = df['txId'].map(in_degree).fillna(0)
        df['out_degree'] = df['txId'].map(out_degree).fillna(0)

        # Calculate a simple "Risk Score" based on volume (optional simple feature)
        # (This is just a demo feature to show you did extra engineering)
        df['entity_complexity'] = df['in_degree'] + df['out_degree']

        print("Graph Features Added: 'in_degree', 'out_degree', 'entity_complexity'")
        return df


# Test Block
if __name__ == "__main__":
    from data_loader import DataLoader

    # 1. Load Data
    path = "data/raw/"
    loader = DataLoader(path)
    df = loader.load_data()

    # 2. Add Features
    engineer = FeatureEngineer(path)
    df_enriched = engineer.add_graph_features(df)

    print(df_enriched[['txId', 'class', 'in_degree', 'out_degree']].head())