# ray-search

Simple library using a builder pattern to solve for top-k search and building text indexes in parallel across multiple nodes. 
It uses apache ray under the hood and is primarily designed to run on databricks clusters. It is a work in progress 
and currently experimental. The apis will break. Please use at your own peril.

## Quick install

```shell
pip install git+https://github.com/stikkireddy/ray-search.git
```

## Start long term ray session locally to access dashboard and test locally

```sh
ray start \
      --head \
      --port=6379 \
      --disable-usage-stats \
      --object-manager-port=8076 \
      --include-dashboard=true \
      --dashboard-host=0.0.0.0 \
      --dashboard-port=8266
ray start --head --include-dashboard=true --disable-usage-stats --dashboard-port 9090 --node-ip-address=0.0.0.0
```

```python
import pandas as pd

from ray_search import *

data = pd.read_csv("data/luggage_review.csv")
ids = data['review_id'].values.astype('U').tolist()
texts = data['review_body_trunc'].values.astype('U').tolist()

worker_ct = 5

matrix_builder: SearchMatrixBuilder = SearchMatrixBuilder() \
        .with_content(ids,
                      texts) \
        .with_workers(worker_ct) \
        .with_worker_memory(Memory.in_mb(512)) \
        .as_512_chunk() \
        .with_vectorizer(Vectorizers.SentenceTransformerDense)

search_builder: SearchBuilder = SearchBuilder() \
      .with_workers(worker_ct) \
      .with_searcher(Searchers.FaissANNSearch) \
      .with_worker_memory(Memory.in_mb(2046)) \
      .with_chunk_size(16) \
      .with_top_k_per_entity(5)
  
results = SearchPipeline() \
      .with_search_matrix_builder(matrix_builder) \
      .with_search_builder(search_builder) \
      .to_df_unnested()

print(results.head(10))
```

## Roadmap

There are three core components that this library offers. Ability to vectorize your code in parallel using the state of the art.
It helps you then run distributed top-k search, given a set of constraints. It also provides sinks for the vectors to be saved 
in a given format/index to disk.

### Vectorizers - Creating dense/sparse vectors 

- [x] Splade sparse vector support creation
- [x] TF-IDF vectorizer creation (single actor)
- [ ] Distributed Count Vectorizer TF-IDF transformer creation (multi actor)
  - can be achieved by generating distributed CSR matrices and merging them
- [ ] Hybrid vectors (Splade sparse + sentencetransformer dense vectors) afaik pinecone is the only one which supports this
- [x] Dense vectors with sentence transformers
- [ ] Huggingface embedding creation 
- [x] Custom Vectorizer (single actor)
- [x] Custom Chunked Vectorizer (multi actor)

### Search - Creating dense sparse vectors 

- [x] Brute force compute Cosine Top K
- [x] Faiss Top K with IndexFlatIP index
- [ ] Custom faiss index strategy via func
  - right now the index isn't built and distributed
- [ ] Annoy Top k
- [ ] MinHash Lsh, MinHash Lsh Forest, MinHash Lsh Ensemble for k ANN search

### Sink - Write the vectorized output to some persistent storage

- [x] Build IndexFlatIP index and write to file 
  - [ ] Move to storage abstraction, separate format from storage target
- [x] Build Chroma Index using duckdb+parquet and local persistent storage
  - [ ] Move to storage abstraction, separate format from storage target
- [ ] Abstraction for materializing the id matrix, index, etc. to storage
  - [ ] Local Filesystem
  - [ ] DBFS databricks file system
  - [ ] S3
  - [ ] ADLS
  - [ ] GCS
- [ ] Llama index support
- [ ] Pinecone support


## Note: Experimental

This library is a work in progress the api will break. Please do not use for production.