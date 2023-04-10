import abc
from typing import Optional, List, Type, Union

import ray

from ray_search.actors import VectorizerActor, SearchActor, Searchers, Vectorizers, is_vectorizer_func, \
    SetupFuncType, IngestionFuncType, get_wrapped_func_attr
from ray_search.index import MatrixWithIds, DistanceThreshold
from ray_search.sinks import SinkConfiguration
from ray_search.utils import window_iter, Memory, SearchResult


class Runner(abc.ABC):

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def _run(self):
        pass


class RayChunkedActorBuilder(Runner):

    def __init__(self):
        self._num_workers = 4  # magic number
        self._chunk_size = 128  # magic number multiple of 2 try anywhere from 128 to 1024
        self._limit = None
        self._cpus_per_worker = 1
        self._memory_per_worker: Memory = Memory.from_gb(1)
        self._actor_class = None

    def with_workers(self, workers: int) -> 'RayChunkedActorBuilder':
        self._num_workers = workers
        return self

    def with_worker_cpu(self, cpus: int):
        self._cpus_per_worker = cpus
        return self

    def with_worker_memory(self, memory: Memory):
        assert isinstance(memory, Memory), "Make sure you use memory instance; eg Memory.from_gb(2)"
        self._memory_per_worker = memory
        return self

    def with_chunk_size(self, chunk_size: int) -> 'RayChunkedActorBuilder':
        assert chunk_size % 2 == 0, "Chunk size must be multiple of 2, try 128 or 256 or 1024"
        self._chunk_size = chunk_size
        return self

    def as_64_chunk(self) -> 'RayChunkedActorBuilder':
        self.with_chunk_size(64)
        return self

    def as_256_chunk(self) -> 'RayChunkedActorBuilder':
        self.with_chunk_size(256)
        return self

    def as_512_chunk(self) -> 'RayChunkedActorBuilder':
        self.with_chunk_size(512)
        return self

    def as_1024_chunk(self) -> 'RayChunkedActorBuilder':
        self.with_chunk_size(1024)
        return self

    def with_limit(self, limit: int) -> 'RayChunkedActorBuilder':
        assert limit > 0, "Limit must be greater than 0."
        self._limit = limit
        return self

    @property
    def actor_class(self):
        return self._actor_class

    @abc.abstractmethod
    def validate(self):
        pass

    @abc.abstractmethod
    def _run(self):
        pass


class SearchPipeline(Runner):

    def __init__(self):
        self._search_matrix_builder: Optional[Union['SearchMatrixBuilder', Runner]] = None
        self._search_builder: Optional[Union['SearchBuilder', Runner]] = None

    def with_search_matrix_builder(self, search_matrix_builder: 'SearchMatrixBuilder') -> 'SearchPipeline':
        self._search_matrix_builder = search_matrix_builder
        return self

    def with_search_builder(self, search_builder: 'SearchBuilder') -> 'SearchPipeline':
        self._search_builder = search_builder
        return self

    def validate(self):
        assert self._search_matrix_builder is not None, "search_matrix_builder must be provided, " \
                                                        "please use with_search_matrix_builder"
        assert self._search_builder is not None, "search_builder must be provided, please use with_search_builder(...)"
        compatible, error_msg = Searchers.is_compatible(self._search_builder.actor_class,
                                                        self._search_matrix_builder.actor_class)
        assert compatible is True, error_msg

        if self._search_matrix_builder.is_producing_dense_vectors() is False:
            assert self._search_builder.actor_class not in [
                Searchers.FaissANNSearch
            ], "Sparse vector search is not supported by FaissANNSearch please create dense vectors"

        self._search_matrix_builder.validate()
        self._search_builder.validate(via_pipeline=True)

    def _run(self):
        self.validate()
        matrix = self._search_matrix_builder._run()
        return self._search_builder.with_matrix(matrix)._run()

    def to_results(self):
        return self._run()

    def to_df(self):
        return SearchResult.to_df(self._run())

    def to_df_unnested(self):
        return SearchResult.to_df(self._run(), unnest=True)


class SearchBuilder(RayChunkedActorBuilder):

    def __init__(self):
        super().__init__()
        self._matrix: Optional[MatrixWithIds] = None
        self._top_k_per_entity: Optional[int] = None
        self._distance_threshold: Optional[DistanceThreshold] = None
        self._with_gc: bool = True
        self._nested = False

    def with_gc(self, do_gc: bool) -> 'SearchBuilder':
        self._with_gc = do_gc
        return self

    def with_top_k_per_entity(self, top_k_per_entity: int) -> 'SearchBuilder':
        self._top_k_per_entity = top_k_per_entity
        return self

    def with_distance_threshold(self, distance_threshold: DistanceThreshold) -> 'SearchBuilder':
        self._distance_threshold = distance_threshold
        return self

    def with_matrix(self, matrix: MatrixWithIds) -> 'SearchBuilder':
        self._matrix = matrix
        return self

    def as_nested(self):
        self._nested = True
        return self

    def with_searcher(self, actor: Type[SearchActor]):
        self._actor_class = actor
        return self

    def validate(self, via_pipeline: bool = False):
        if via_pipeline is False:  # we want to skip this during pipeline since we wont have matrix
            assert self._matrix is not None, "Matrix must be provided"
        assert self._actor_class is not None, "Search strategy must be provided, please use with_searcher(...)"

    def _run(self):
        ray_actor = ray.remote(num_cpus=self._cpus_per_worker, memory=self._memory_per_worker.bytes)(self._actor_class)
        matrix_remote = ray.put(self._matrix)  # only need one instance of this object
        search_actors = [ray_actor.remote(matrix_remote) for _ in range(self._num_workers)]
        results = []
        for idx, window in enumerate(window_iter(self._limit or self._matrix.num_rows(), self._chunk_size)):
            results.append(
                search_actors[idx % len(search_actors)]
                .search.remote(
                    self._matrix[window.start:window.end],
                    top_k_per_entity=self._top_k_per_entity,
                    distance_threshold=self._distance_threshold,
                    with_gc=self._with_gc
                )
            )

        # unpack list of lists with list comprehension
        return [res for res_list in ray.get(results) for res in res_list]


class SearchMatrixBuilder(RayChunkedActorBuilder):

    def __init__(self):
        super().__init__()
        self._ids = None
        self._texts = None
        self._setup_hook = None
        self._ingest_content_chunk_func = None

    def with_content(self, ids: List[str], texts: List[str]) -> 'SearchMatrixBuilder':
        assert len(ids) == len(texts), "Make sure id list and text list is equal"
        self._ids = ids
        self._texts = texts
        return self

    def is_producing_dense_vectors(self) -> bool:
        if self.actor_class is not None:
            self.actor_class: Type[VectorizerActor]
            return self.actor_class.is_dense() or \
                get_wrapped_func_attr(self._ingest_content_chunk_func, "produces_dense_vectors")
        return False

    def with_vectorizer(self, index_class: Type[VectorizerActor]):
        self._actor_class = index_class
        return self

    def validate(self):
        assert self._ids is not None and self._texts is not None, "Must have ids and texts"
        assert self._actor_class is not None, "Must choose a deployable actor"
        if self._actor_class in [Vectorizers.CustomFunc, Vectorizers.CustomChunkedFunc]:
            assert self._setup_hook is not None and self._ingest_content_chunk_func is not None, \
                f"For: {Vectorizers.CustomFunc} and {Vectorizers.CustomChunkedFunc} you must provide, " \
                f"with_setup_hook and with_ingest_func"
            assert is_vectorizer_func(self._ingest_content_chunk_func), "Please use the vectorizer decorator"

    def with_setup_hook(self, setup_hook: Optional[SetupFuncType]) -> 'SearchMatrixBuilder':
        self._setup_hook = setup_hook
        return self

    def with_ingest_func(self, ingest_content_chunk_func: Optional[IngestionFuncType]) \
            -> 'SearchMatrixBuilder':
        self._ingest_content_chunk_func = ingest_content_chunk_func
        return self

    def make_content_map(self):
        return dict(zip(self._ids, self._texts))

    def _run(self) -> MatrixWithIds:
        # print(f"Running {self}")
        # make actors dynamically, and if limit is there do not more actors than the limit
        ray_actor = ray.remote(num_cpus=self._cpus_per_worker, memory=self._memory_per_worker.bytes)(self._actor_class)
        actor_args = [self._setup_hook, self._ingest_content_chunk_func]
        if self._actor_class.supports_chunking() is False:
            splade_actors = [ray_actor.remote(*actor_args)]
            if self._limit is not None:
                splade_actors[0].ingest_texts.remote(self._ids[0:self._limit], self._texts[0:self._limit])
            else:
                splade_actors[0].ingest_texts.remote(self._ids, self._texts)
        else:
            num_actors = self._num_workers if self._limit is None else min(self._num_workers, self._limit)
            num_actors = min(num_actors, len(self._ids))
            splade_actors = [ray_actor.remote(*actor_args) for _ in range(num_actors)]
            for idx, window in enumerate(window_iter(self._limit or len(self._ids), self._chunk_size)):
                ids = self._ids[window.start:window.end]
                texts = self._texts[window.start:window.end]
                splade_actors[idx % len(splade_actors)].ingest_texts.remote(ids, texts)
        return MatrixWithIds.merge(
            *[ray.get(actor.get_matrix.remote()) for actor in splade_actors]
        )

    def to_sink(self, sink_klass: SinkConfiguration):
        self.validate()
        sink_klass.validate()
        matrix = self._run()
        sink = sink_klass.to_sink()
        # TODO: create a sink result object
        sink.consume(matrix)
        return sink


class Builders:
    BuildSearchMatrix = SearchMatrixBuilder
    BuildSearch = SearchBuilder
    BuildPipeline = SearchPipeline
