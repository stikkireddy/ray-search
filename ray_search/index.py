import abc
import gc
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics.pairwise import cosine_distances

from ray_search.utils import SearchResult, window_iter


class BaseSearchIndex(abc.ABC):

    def __init__(self, id_map: Dict[str, int], index_map: Dict[int, str]):
        self.id_map: Dict[str, int] = id_map
        self.index_map: Dict[int, str] = index_map


class MatrixWithIds(BaseSearchIndex):

    def __init__(self, matrix: Union[csr_matrix, np.matrix, np.ndarray],
                 id_map: Dict[str, int], index_map: Dict[int, str]):
        super().__init__(id_map, index_map)
        self.matrix: Union[csr_matrix, np.matrix, np.ndarray] = matrix

    def add_matrix(self, ids: List[str], matrix: Union[csr_matrix, np.matrix, np.ndarray]):
        assert len(ids) == matrix.shape[0], "Matrix and list of ids should be the same"
        assert ids is not None, "Ids must exist"
        assert matrix is not None, "sparse matrix must exist"
        if isinstance(matrix, np.matrix):
            matrix = np.asarray(matrix)
        if self.matrix is None:
            self.matrix = matrix
            # todo: bidirectional look up id to index and index to id
            self.id_map = {id_: idx for idx, id_ in enumerate(ids)}
            self.index_map = {idx: id_ for idx, id_ in enumerate(ids)}
            return
        current_index = self.matrix.shape[0]  # index already includes + 1
        for i in ids:
            assert i not in self.id_map, f"Id: {i} already exists in map"
        self.id_map.update({id_: idx + current_index for idx, id_ in enumerate(ids)})
        self.index_map.update({idx + current_index: id_ for idx, id_ in enumerate(ids)})
        self.matrix = self.stack([self.matrix, matrix])

    def stack(self, matrix_list: List[Union[np.matrix, csr_matrix]]):
        if isinstance(self.matrix, csr_matrix):
            return vstack(matrix_list)
        if isinstance(self.matrix, np.matrix):
            return np.vstack(matrix_list)
        if isinstance(self.matrix, np.ndarray):
            return np.vstack(matrix_list)

    def index_list(self):
        return list(self.index_map.keys())

    def consume_pytorch_tensors(self, ids: List[str], tensors: torch.Tensor, with_gc: bool = False):
        self.add_matrix(ids, csr_matrix(tensors.numpy(), copy=True))
        del tensors
        if with_gc is True:
            gc.collect(generation=2)

    def approx_size_gb(self):
        self.approx_size_bytes() / (1024.0 ** 3)

    def approx_size_mb(self):
        self.approx_size_bytes() / (1024.0 ** 2)

    def approx_size_bytes(self):
        if isinstance(self.matrix, csr_matrix):
            return (self.matrix.data.nbytes + self.matrix.indptr.nbytes + self.matrix.indices.nbytes +
                    sys.getsizeof(self.id_map) + sys.getsizeof(self.index_map))
        elif isinstance(self.matrix, np.matrix):
            return (self.matrix.size * self.matrix.itemsize +
                    sys.getsizeof(self.id_map) + sys.getsizeof(self.index_map))
        elif isinstance(self.matrix, np.ndarray):
            return (self.matrix.size * self.matrix.itemsize +
                    sys.getsizeof(self.id_map) + sys.getsizeof(self.index_map))

    def get_slice_by_id(self, id_: str) -> Optional[csr_matrix]:
        index = self.id_map.get(id_, None)
        if index is None:
            return None
        return self.matrix[index]

    def num_rows(self):
        return self.matrix.shape[0]

    def num_cols(self):
        return self.matrix.shape[1]

    def get_id(self, index):
        return self.index_map[index]

    def get_matrix_index(self, id_):
        return self.id_map[id_]

    def iter(self, page_size=100):
        for w in window_iter(self.num_rows(), page_size):
            yield self[w.start:w.end]

    def __getitem__(self, item):
        if isinstance(item, int):
            id_ = self.index_map[item]
            return MatrixWithIds(self.matrix[item], {id_: 0}, {0: id_})  # one item will always index at 0
        if isinstance(item, slice):
            assert item.step is None, "Slicing with steps is not supported. Please pick contiguous sections"
            ids = [self.index_map[i] for i in range(item.start, item.stop)]
            return MatrixWithIds(
                self.matrix[item.start:item.stop],
                {id_: idx for idx, id_ in enumerate(ids)},
                {idx: id_ for idx, id_ in enumerate(ids)},
            )

    def __str__(self) -> str:
        return f"""{repr(self.matrix)}, id_map keys: {len(self.id_map)}, index_map keys: {len(self.index_map)}"""

    @classmethod
    def from_empty(cls):
        return cls(None, {}, {})

    @staticmethod
    def merge(*matrices: 'MatrixWithIds') -> 'MatrixWithIds':
        filtered_matrices = [m for m in matrices if m.matrix is not None]
        first_matrix = filtered_matrices[0]
        if len(matrices) == 1:
            return first_matrix
        for matrix_with_ids in filtered_matrices[1:]:
            ids_ = [matrix_with_ids.index_map[index] for index in range(len(matrix_with_ids.index_map))]
            first_matrix.add_matrix(ids_, matrix_with_ids.matrix)
        return first_matrix


def build_faiss_index(matrix: MatrixWithIds) -> 'faiss.IndexIDMap':
    import faiss
    id_index = np.array(matrix.index_list()).astype('int')
    content_normalized = matrix.matrix.copy()
    faiss.normalize_L2(content_normalized)
    idx = faiss.IndexIDMap(faiss.IndexFlatIP(matrix.num_cols()))
    idx.add_with_ids(content_normalized, id_index)
    del content_normalized
    return idx


def query_faiss_index(
        top_k_per_entity: int,
        input_ids: List[str],
        input_matrix: Union[csr_matrix, np.matrix, np.ndarray],
        index: 'faiss.IndexIDMap',
        idx_id_map: Dict[int, str],
        with_gc: bool = True
) -> List[SearchResult]:
    import faiss
    query_matrix_normalized = input_matrix.copy()
    faiss.normalize_L2(query_matrix_normalized)
    top_k = index.search(query_matrix_normalized, top_k_per_entity + 1)
    ids = np.vectorize(idx_id_map.get)(top_k[1])
    results = []
    for idx, scores in enumerate(top_k[0]):
        current_input_id = input_ids[idx]
        results.append(
            SearchResult(
                input_id=current_input_id,
                search_results=[{"search_result_id": id_,
                                 "score": score}
                                for id_, score in zip(ids[idx], scores) if id_ != current_input_id]
            )
        )
    if with_gc is True:
        del query_matrix_normalized
        gc.collect(generation=2)
    return results


@dataclass
class DistanceThreshold:
    # only one of the three will be executed in order and inclusive
    greater_than: Optional[float] = None
    less_than: Optional[float] = None
    between: Optional[Tuple[float, float]] = None

    def index_of(self, arr: np.array) -> List[int]:
        if self.greater_than is not None:
            return np.where((arr >= self.greater_than))[0].tolist()
        if self.less_than is not None:
            return np.where((arr <= self.greater_than))[0].tolist()
        if self.between is not None:
            return np.where((arr >= self.between[0]) & (arr <= self.between[1]))[0].tolist()


def calculate_cosine_distance(
        input_matrix: MatrixWithIds,
        target_matrix: MatrixWithIds = None,
        top_k_per_entity: Optional[int] = None,
        distance_threshold: Optional[DistanceThreshold] = None,
        with_gc: bool = False) -> List[SearchResult]:
    if target_matrix is None:
        target_matrix = input_matrix
    # matches = {}
    matches = []
    input_data = input_matrix.matrix
    target_data = target_matrix.matrix
    distances = 1 - cosine_distances(input_data, target_data)
    for idx, distance in enumerate(distances):
        if top_k_per_entity is not None:
            indexes = np.argpartition(distance, -1 * (top_k_per_entity + 1))[-1 * (top_k_per_entity + 1):]
        elif distance_threshold is not None:
            indexes = distance_threshold.index_of(distance)
        else:
            indexes = range(0, len(distance))

        current_id = input_matrix.get_id(idx)
        matches.append(
            # this may make things slower :| but easier to write code wish this was rust :-)
            SearchResult(
                input_id=current_id,
                search_results=[{"search_result_id": target_matrix.get_id(index),
                                 "score": distance[index]} for index in indexes
                                if target_matrix.get_id(index) != current_id]
            )
        )

    del distances
    if with_gc is True:
        gc.collect(generation=2)

    return matches

# if __name__ == "__main__":
#     from sentence_transformers import SentenceTransformer
#
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     matrix = MatrixWithIds.from_empty()
#     # Sentences we want to encode. Example:
#     # sentence = [
#     #     'This framework generates embeddings for each input sentence',
#     #     "This framework is generating another sentence"
#     # ]
#     data = pd.read_csv("../data/luggage_review.csv")
#     ids = data['review_id'].values.astype('U').tolist()[0:100]
#     texts = data['review_body_trunc'].values.astype('U').tolist()[0:100]
#
#     # Sentences are encoded by calling model.encode()
#     embedding = model.encode(texts)
#     # print(embedding)
#     # print(type(embedding))
#
#     matrix.add_matrix(ids, embedding)
#     matrix.add_matrix([f"{i}-{i}" for i in ids], embedding)
#     # matrix.add_matrix(["id3", "id4"], embedding)
#     print(matrix.matrix.shape)
#     print(matrix.num_cols())
#     print(matrix.approx_size_bytes())
