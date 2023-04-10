import abc
from pathlib import Path
from typing import Any, Type, List

import numpy as np
from sentence_transformers import SentenceTransformer

from ray_search.index import MatrixWithIds, build_faiss_index, query_faiss_index


class SinkConfiguration(abc.ABC):

    def __init__(self, sink_type: Type['Self']):
        self._sink_type: Type['Self'] = sink_type

    @abc.abstractmethod
    def _validate(self) -> (bool, str):
        pass

    @abc.abstractmethod
    def _to_sink(self) -> 'Sink':
        pass

    def to_sink(self) -> 'Sink':
        return self._to_sink()

    def validate(self):
        supports, error = self._sink_type.supports_configuration(self.__class__)
        assert supports is True, error
        self._validate()


class Sink(abc.ABC):

    @abc.abstractmethod
    def consume(self, matrix: MatrixWithIds) -> Any:
        pass

    def to_bytes(self):
        import cloudpickle
        return cloudpickle.dumps(self)

    @staticmethod
    def from_bytes(bytes_) -> 'Self':
        import cloudpickle
        return cloudpickle.loads(bytes_)

    @staticmethod
    @abc.abstractmethod
    def supports_configuration(sink_config_type: Type['SinkConfiguration']) -> (bool, str):
        pass


class DefaultSinkConfiguration(SinkConfiguration):

    def _validate(self):
        return True, None

    def _to_sink(self) -> Sink:
        return self._sink_type()


class LocalFileSinkConfiguration(SinkConfiguration):

    def __init__(self, sink_type: Type['FaissIndex']):
        super().__init__(sink_type)
        self._local_file_path = None

    def _validate(self) -> (bool, str):
        assert self._local_file_path is not None, "Must provide file path, use with_local_file_path"

    def _to_sink(self) -> 'Sink':
        return self._sink_type(self._local_file_path)

    def with_local_file_path(self, file_path: str) -> 'LocalFileSinkConfiguration':
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local_file_path = file_path
        return self


class FaissIndex(Sink):

    @staticmethod
    def supports_configuration(sink_config_type: Type['DefaultSinkConfiguration']) -> (bool, str):
        return sink_config_type in [
            DefaultSinkConfiguration,
            LocalFileSinkConfiguration
        ], "Only supports DefaultSinkConfiguration, LocalFileSinkConfiguration"

    def __init__(self, save_to_local_path=None):
        self._save_to_local_path = save_to_local_path
        self._index = None
        self._index_id_map = None

    @property
    def index(self):
        return self._index

    @property
    def index_id_map(self):
        return self._index_id_map

    def query(self,
              model: SentenceTransformer,
              ids: List[str],
              texts: List[str],
              top_k_per_entity: int = 5,
              with_gc=False):
        input_matrix: np.ndarray = model.encode(texts)
        return query_faiss_index(
            top_k_per_entity,
            input_ids=ids,
            input_matrix=input_matrix,
            index=self._index,
            idx_id_map=self._index_id_map,
            with_gc=with_gc
        )

    def consume(self, matrix: MatrixWithIds):
        self._index = build_faiss_index(matrix)
        self._index_id_map = matrix.index_map

        if self._save_to_local_path is not None:
            with Path(self._save_to_local_path).open("wb") as f:
                f.write(self.to_bytes())


class ChromaSinkConfiguration(SinkConfiguration):

    def __init__(self, sink_type: Type['ChromaSink']):
        super().__init__(sink_type)
        self._ingest_chunk_size = 1000
        self._chroma_db_impl = "duckdb+parquet"
        self._directory = None
        self._collection_name = None

    def with_add_chunk_size(self, add_chunk_size: int):
        self._ingest_chunk_size = add_chunk_size
        return self

    def with_db_dir(self, directory: str):
        self._directory = directory
        return self

    def with_collection_name(self, collection_name: int):
        self._collection_name = collection_name
        return self

    def with_chroma_db_impl(self, chroma_db_impl: str):
        self._chroma_db_impl = chroma_db_impl
        return self

    def _validate(self) -> (bool, str):
        assert self._collection_name is not None, "Must provide collection Name, with_collection_name"
        assert self._directory is not None, "Must provide directory name, with_db_dir"

    def _to_sink(self) -> Sink:
        return self._sink_type(self._collection_name,
                               self._directory,
                               self._chroma_db_impl,
                               self._ingest_chunk_size)


class ChromaSink(Sink):

    @staticmethod
    def supports_configuration(sink_config_type: Type['DefaultSinkConfiguration']) -> (bool, str):
        return sink_config_type == ChromaSinkConfiguration, "Only supports ChromaSinkConfiguration"

    def __init__(self,
                 collection_name,
                 directory,
                 chroma_db_impl: str = "duckdb+parquet",
                 ingest_chunk_size: int = 100):
        self._ingest_chunk_size = ingest_chunk_size
        self._index = None
        self._collection_name = collection_name
        self._directory = directory
        self._chroma_db_impl = chroma_db_impl

    def client(self):
        import chromadb
        from chromadb.config import Settings
        return chromadb.Client(Settings(
            chroma_db_impl=self._chroma_db_impl,
            persist_directory=self._directory
        ))

    @property
    def index(self):
        return self.client().get_or_create_collection(self._collection_name)

    def consume(self, matrix: MatrixWithIds):
        client = self.client()
        coll = client.get_or_create_collection(self._collection_name)
        for idx, chunk in enumerate(matrix.iter(self._ingest_chunk_size)):
            print(f"Started chroma sink chunk: {idx}")
            coll.add(
                embeddings=chunk.matrix.tolist(),
                ids=[chunk.get_id(i) for i in range(self._ingest_chunk_size)]
            )
            print(f"Finished chroma sink chunk: {idx}")


class Sinks:
    Faiss = DefaultSinkConfiguration(FaissIndex)
    FaissFileSink = LocalFileSinkConfiguration(FaissIndex)
    ChromaSinkBuilder = ChromaSinkConfiguration(ChromaSink)
