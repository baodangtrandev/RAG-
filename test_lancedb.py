import lancedb
import time
import numpy as np

db = lancedb.connect('/tmp/lancedb')
table = db.open_table('github')

# Dummy query
emb = np.random.rand(1024).astype(np.float32)

start = time.time()
res = table.search(emb).limit(10).to_list()
print("Search time default (L2):", time.time() - start)

start = time.time()
res = table.search(emb).metric("cosine").limit(10).to_list()
print("Search time cosine:", time.time() - start)

print("Indices:", table.list_indices())
