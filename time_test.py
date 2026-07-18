import time
import lancedb
import numpy as np

start = time.time()
db = lancedb.connect('/tmp/lancedb')
print('Connect DB:', time.time() - start)

table = db.open_table('confluence')
print('Open Table:', time.time() - start)

dummy_emb = np.random.rand(1024).astype(np.float32)

start_search = time.time()
res = table.search(dummy_emb).limit(20).to_list()
print('Vector search (Flat):', time.time() - start_search)
