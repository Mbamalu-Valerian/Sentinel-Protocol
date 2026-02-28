import pandas as pd
import time
import os
import random
import shutil

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
LIVE_BUFFER_PATH = os.path.join(BASE_DIR, 'data', 'live_buffer.csv')


def start_streaming():
    print("-------------------------------------------------")
    print("🚀 STARTING MOCK BLOCKCHAIN NODE...")
    print("📡 CONNECTING TO HISTORICAL LEDGER REPLAY...")
    print("-------------------------------------------------")

    # 1. Load the "Unseen" Test Data (Transactions the model hasn't learned)
    # We cheat slightly by loading the raw features to simulate incoming packets
    features = pd.read_csv(os.path.join(RAW_DATA_PATH, 'elliptic_txs_features.csv'), header=None)
    classes = pd.read_csv(os.path.join(RAW_DATA_PATH, 'elliptic_txs_classes.csv'))

    # Rename for clarity
    features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(165)]

    # Merge to get the class (so we know if we are streaming fraud or not)
    # In a real system, we wouldn't have the class, but we need it here to SHOW the demo works
    full_data = pd.merge(features, classes, on='txId')

    # Filter for 'unknown' class to simulate raw network data,
    # OR filter for known Fraud/Licit to make the demo exciting.
    # Let's pick a mix of Fraud (1) and Licit (2) to stream.
    stream_pool = full_data[full_data['class'] != 'unknown']

    # Create the buffer file with headers
    if not os.path.exists(LIVE_BUFFER_PATH):
        with open(LIVE_BUFFER_PATH, 'w') as f:
            f.write("txId,status,timestamp\n")

    print("🟢 NODE ONLINE. Broadcasting transactions...")

    while True:
        # 1. Pick a random transaction
        tx = stream_pool.sample(1).iloc[0]
        tx_id = int(tx['txId'])
        tx_class = "FRAUD" if tx['class'] == '1' else "LICIT"

        # 2. "Broadcast" it to the system (Write to a buffer file)
        # The Dashboard will read this file
        with open(LIVE_BUFFER_PATH, 'a') as f:
            # We just log the ID here. The dashboard uses the ID to look up the features.
            # In a real real system, we would write the full feature vector.
            f.write(f"{tx_id},{tx_class},{time.time()}\n")

        print(f"⚡ [NEW BLOCK] TxID: {tx_id} | Type: {tx_class} | Broadcasted to Network.")

        # 3. Wait a random amount of time (0.5 to 2 seconds) to simulate network latency
        time.sleep(random.uniform(0.5, 2.0))


if __name__ == "__main__":
    start_streaming()