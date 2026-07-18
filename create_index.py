import lancedb
import time
from lancedb.index import IvfPq

db = lancedb.connect('/tmp/lancedb')
tables = db.table_names()

for t in tables:
    print(f"Creating index for {t}...", flush=True)
    start = time.time()
    table = db.open_table(t)
    try:
        config = IvfPq(
            distance_type="l2",
            num_partitions=64,
            num_sub_vectors=64,
            max_iterations=10
        )
        # Fix: pass "vector" as the first positional argument
        table.create_index(
            "vector",
            config=config,
            replace=True
        )
        print(f"Done {t} in {time.time() - start:.2f}s", flush=True)
    except Exception as e:
        print(f"Failed {t}: {e}", flush=True)
