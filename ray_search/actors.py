import abc
import functools
import gc
import os
from typing import Optional, List, Type, Any, Callable

import faiss
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ray_search.index import MatrixWithIds, DistanceThreshold, calculate_cosine_distance, build_faiss_index, \
    query_faiss_index
from ray_search.utils import window_iter, SearchResult


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


IngestionFuncType = Callable[['Self', List[str], List[str]], None]

SetupFuncType = Callable[['Self'], None]


class VectorizerActor(abc.ABC):

    def __init__(self,
                 setup_hook: Optional[SetupFuncType] = None,
                 ingest_content_chunk_func: Optional[IngestionFuncType] = None
                 ):
        self.matrix_with_ids = MatrixWithIds.from_empty()
        self.ingest_chunk_size = None
        self._ingest_text_chunk_func = ingest_content_chunk_func

        self.setup_state = {}
        if setup_hook is not None:
            setup_hook(self)

    def get_matrix(self) -> MatrixWithIds:
        return self.matrix_with_ids

    def set_state(self, key: str, value: Any):
        self.setup_state[key] = value

    def get_state(self, key: str) -> Optional[Any]:
        return self.setup_state.get(key, None)

    def ingest_texts(self, ids: List[str], texts: List[str]) -> None:
        if self.ingest_chunk_size is None:
            self._ingest_text_chunk(ids, texts)
            return

        for w in window_iter(len(texts), self.ingest_chunk_size):
            self._ingest_text_chunk(ids[w.start:w.end], texts[w.start:w.end])

    @abc.abstractmethod
    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def supports_chunking():
        pass

    @staticmethod
    def is_dense() -> Optional[bool]:
        return None


# @ray.remote(num_cpus=1, memory=1024 * 1024 * 1024 * 1)
class SpladeVectorizer(VectorizerActor):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = int(os.environ.get("ITERATOR_CHUNK_SIZE" , "32"))  # overwrite chunksize
        self.model_id = os.environ.get("SPLADE_MODEL_NAME", 'naver/splade-cocondenser-ensembledistil')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)

    def _ingest_text_chunk(self, chunk_ids: List[str], chunk_texts: List[str]) -> None:
        tokens = self.tokenizer(
            chunk_texts, return_tensors='pt',
            padding=True, truncation=True
        )
        # print(tokens)
        output = self.model(**tokens)
        # print(output)
        # aggregate the token-level vecs and transform to sparse
        vecs = torch.max(
            torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), dim=1
        )[0].detach().cpu()

        self.matrix_with_ids.consume_pytorch_tensors(chunk_ids, vecs)
        del vecs

    @staticmethod
    def supports_chunking():
        return True

    @staticmethod
    def is_dense() -> Optional[bool]:
        return False


class TFIDFSparseVectorizer(VectorizerActor):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = None

    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        v = TfidfVectorizer()
        matrix = v.fit_transform(texts)
        self.matrix_with_ids.add_matrix(ids, matrix)

    @staticmethod
    def supports_chunking():
        return False

    @staticmethod
    def is_dense() -> Optional[bool]:
        return False


class SentenceTransformerDenseVectorizer(VectorizerActor):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = int(os.environ.get("ITERATOR_CHUNK_SIZE" , "32"))  # construction will always yield 0
        self.model = self.transformer_model()

    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        embedding = np.asmatrix(self.model.encode(texts))
        self.matrix_with_ids.add_matrix(ids, embedding)
        del embedding

    @staticmethod
    def transformer_model():
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("SENTENCE_TRANSFORMER_MODEL_NAME", "all-MiniLM-L6-v2")
        return SentenceTransformer(model_name)

    @staticmethod
    def supports_chunking():
        return True

    @staticmethod
    def is_dense() -> Optional[bool]:
        return True


class CustomFunctionVectorizer(VectorizerActor):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = None

    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        self._ingest_text_chunk_func(self, ids, texts)

    @staticmethod
    def supports_chunking():
        return False


class CustomFunctionChunkedVectorizer(CustomFunctionVectorizer):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = get_wrapped_func_attr(self._ingest_text_chunk_func, "chunk_size") or int(os.environ.get("ITERATOR_CHUNK_SIZE" , "32"))

    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        self._ingest_text_chunk_func(self, ids, texts)

    @staticmethod
    def supports_chunking():
        return True


def vectorizer(*, produces_dense_vectors: bool, chunk_size: int = 32):
    def wrapper(f):
        @functools.wraps(f)
        def decorated_func(*args, **kwargs):
            return f(*args, **kwargs)

        f.vectorizer_func = True
        f.chunk_size = chunk_size
        f.produces_dense_vectors = produces_dense_vectors
        return decorated_func

    return wrapper


def get_wrapped_func_attr(func: Callable, attr: str) -> Optional[Any]:
    if "__wrapped__" in func.__dict__ and attr in func.__dict__["__wrapped__"].__dict__:
        return func.__dict__["__wrapped__"].__dict__[attr]

    return None


def is_vectorizer_func(func: IngestionFuncType) -> bool:
    return get_wrapped_func_attr(func, "vectorizer_func") is True


class Searchers:
    CosineSearch = CosineSearcher
    FaissANNSearch = FaissANNSearcher

    @staticmethod
    def is_compatible(searcher: Type[SearchActor],
                      vectorizer: Type[VectorizerActor]) -> (bool, str):
        if searcher == Searchers.CosineSearch:
            return True, None
        elif vectorizer in [Vectorizers.CustomFunc, Vectorizers.CustomChunkedFunc]:
            return True, None
        elif searcher == Searchers.FaissANNSearch and vectorizer not in [Vectorizers.SentenceTransformerDense]:
            return False, f"{Searchers.FaissANNSearch} is currently only compatible with " \
                          f"{Vectorizers.SentenceTransformerDense}"
        elif searcher == FaissANNSearcher and vectorizer in [Vectorizers.SentenceTransformerDense]:
            return True, None
        return False, f"Unknown searcher: {searcher} or Unknown vectorizer: {vectorizer}"


class Vectorizers:
    SentenceTransformerDense = SentenceTransformerDenseVectorizer
    TFIDFSparse = TFIDFSparseVectorizer
    Splade = SpladeVectorizer
    CustomFunc = CustomFunctionVectorizer
    CustomChunkedFunc = CustomFunctionChunkedVectorizer
