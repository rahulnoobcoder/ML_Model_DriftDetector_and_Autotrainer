import os
import pandas as pd
from filelock import FileLock

DATA_PATH = "data/incoming_data.csv"
LOCK_PATH = "data/incoming_data.csv.lock"

def log_data(df, label=None):
    """
    Logs data to CSV with thread/process safety using a FileLock.
    """
    df = df.copy()
    df["label"] = label

    # The lock ensures only ONE process writes to the file at a time.
    # If another process holds the lock, this line waits (blocks) until it's free.
    lock = FileLock(LOCK_PATH)

    try:
        with lock.acquire(timeout=10):  # Wait up to 10s for the lock
            if os.path.exists(DATA_PATH):
                df.to_csv(DATA_PATH, mode="a", header=False, index=False)
            else:
                df.to_csv(DATA_PATH, index=False)
    except TimeoutError:
        print("ERROR: Could not acquire lock to save data. System is too busy.")