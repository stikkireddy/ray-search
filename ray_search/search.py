import abc
from typing import Optional, List, Type

from ray_search.index import DistanceThreshold, calculate_cosine_distance
from ray_search.index import MatrixWithIds, SearchResult, build_faiss_index, query_faiss_index
from ray_search.vectorizers import Vectorizers, VectorizerConfig


class SearchActor(abc.ABC):

    @abc.abstractmethod
    def search(self,
               input_matrix: MatrixWithIds,
               top_k_per_entity: Optional[int] = None,
               distance_threshold: Optional[DistanceThreshold] = None,
               with_gc: bool = True) -> List[SearchResult]:
        pass


class CosineSearcher(SearchActor):

    def __init__(self, matrix: MatrixWithIds):
        self._matrix = matrix

    def search(self,
               input_matrix: MatrixWithIds,
               top_k_per_entity: Optional[int] = None,
               distance_threshold: Optional[DistanceThreshold] = None,
               with_gc: bool = True) -> List[SearchResult]:
        return calculate_cosine_distance(input_matrix, self._matrix, top_k_per_entity, distance_threshold, with_gc)


class FaissANNSearcher(SearchActor):

    # TODO: support building index once and pass along
    def __init__(self, matrix: MatrixWithIds):
        self._matrix: MatrixWithIds = matrix
        self._index = build_faiss_index(self._matrix)

    def search(self,
               input_matrix: MatrixWithIds,
               top_k_per_entity: Optional[int] = None,
               distance_threshold: Optional[DistanceThreshold] = None,
               with_gc: bool = True) -> List[SearchResult]:
        assert top_k_per_entity is not None, f"{FaissANNSearcher} search requires top_k_per_entity"
        if distance_threshold is not None:
            print(f"{FaissANNSearcher.__name__} ignores distance_threshold")
        return query_faiss_index(
            top_k_per_entity,
            input_ids=[input_matrix.get_id(i) for i in range(input_matrix.num_rows())],
            input_matrix=input_matrix.matrix,
            index=self._index,
            idx_id_map=self._matrix.index_map,
            with_gc=with_gc
        )


class Searchers:
    CosineSearch = CosineSearcher
    FaissANNSearch = FaissANNSearcher

    @staticmethod
    def is_compatible(searcher: Type[SearchActor],
                      vectorizer: Type[VectorizerConfig]) -> (bool, str):
        if searcher == Searchers.CosineSearch:
            return True, None
        elif vectorizer in [Vectorizers.CustomFunc.vectorizer_class,
                            Vectorizers.CustomChunkedFunc.vectorizer_class]:
            return True, None
        elif searcher == Searchers.FaissANNSearch and vectorizer not in \
                [Vectorizers.SentenceTransformerDense.vectorizer_class]:
            return False, f"{Searchers.FaissANNSearch} is currently only compatible with " \
                          f"{Vectorizers.SentenceTransformerDense.vectorizer_class}"
        elif searcher == FaissANNSearcher and vectorizer in [Vectorizers.SentenceTransformerDense.vectorizer_class]:
            return True, None
        return False, f"Unknown searcher: {searcher} or Unknown vectorizer: {vectorizer.vectorizer_class}"
