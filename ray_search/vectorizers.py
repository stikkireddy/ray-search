import abc
import functools
from typing import Callable, List, Optional, Any, Type

import numpy as np
import ray
import torch
from ray.types import ObjectRef
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from ray_search.index import MatrixWithIds
from ray_search.utils import VmType, window_iter, RayCluster

IngestionFuncType = Callable[['Self', List[str], List[str]], None]
SetupFuncType = Callable[['Self'], None]


class VectorizerConfig:

    def __init__(self, vectorizer_klass: Type['VectorizerActor']):
        self._vectorizer_klass = vectorizer_klass
        self._result_chunk_size = None
        self._ray_cluster = None
        self._encoder_batch_size = None
        self._use_gpu = None
        self._setup_hook = None
        self._ingest_content_chunk_func = None
        self._memory_class = None
        self._model_id = None

    @property
    def vectorizer_class(self):
        return self._vectorizer_klass

    def validate(self):
        if self._vectorizer_klass in [Vectorizers.CustomFunc, Vectorizers.CustomChunkedFunc]:
            assert self._setup_hook is not None and self._ingest_content_chunk_func is not None, \
                f"For: {Vectorizers.CustomFunc} and {Vectorizers.CustomChunkedFunc} you must provide, " \
                f"with_setup_hook and with_ingest_func"
            assert is_vectorizer_func(self._ingest_content_chunk_func), "Please use the vectorizer decorator"

    def with_result_chunk_size(self, size: int):
        if self._vectorizer_klass.supports_chunking():
            self._result_chunk_size = size
        return self

    def with_cluster(self, ray_cluster: RayCluster):
        self._ray_cluster = ray_cluster
        return self

    def with_model_id(self, model_id: str):
        self._model_id = model_id
        return self

    def with_encoder_batch_size(self, size: int):
        self._encoder_batch_size = size
        if self._result_chunk_size is None or self._result_chunk_size < size:
            # iterator needs to be atleast of this size for chunking to happen properly
            self._result_chunk_size = size
        return self

    def with_cuda(self):
        self._use_gpu = True
        return self

    def with_memory_class(self, vm_type: VmType):
        self._memory_class = vm_type
        return self

    def with_setup_hook(self, setup_hook: Optional[SetupFuncType]):
        self._setup_hook = setup_hook
        return self

    def with_ingest_func(self, ingest_content_chunk_func: Optional[IngestionFuncType]):
        self._ingest_content_chunk_func = ingest_content_chunk_func
        return self

    def to_vectorizer(self):
        return self._vectorizer_klass(
            self
        )


class VectorizerActor(abc.ABC):

    def __init__(self, config: VectorizerConfig = None):
        self.config = config
        self.matrix_with_ids = MatrixWithIds.from_empty()
        self.ingest_chunk_size = None
        self._ingest_text_chunk_func = self.config._ingest_content_chunk_func
        self.memory_class = self.config._memory_class

        self.setup_state = {}
        if self.config._setup_hook is not None:
            self.config._setup_hook(self)

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

    def get_chunk_size_by_vm_type(self) -> int:
        if self.memory_class is None:
            return 32
        elif self.memory_class is VmType.LOW_MEMORY:
            return 32
        elif self.memory_class is VmType.MED_MEMORY:
            return 64
        elif self.memory_class is VmType.HIGH_MEMORY:
            return 128

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


class MatrixWithIdsFragmentManager:

    @classmethod
    def consume_actors(cls, remote_vectorizer_actors: List[ObjectRef[VectorizerActor]], merge=True):
        unfinished_actors = remote_vectorizer_actors
        matrix_array: List[MatrixWithIds] = []
        while len(unfinished_actors):
            done_id, unfinished_actors = ray.wait(unfinished_actors)
            if merge is True:
                matrix_array.append(ray.get(done_id[0]))

        return MatrixWithIds.merge(*matrix_array)


class SpladeVectorizer(VectorizerActor):

    def __init__(self, *args):
        super().__init__(*args)
        self.ingest_chunk_size = self.config._result_chunk_size or \
                                 self.get_chunk_size_by_vm_type()  # overwrite chunksize
        self.model_id = self.config._model_id or 'naver/splade-cocondenser-ensembledistil'
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
        self.ingest_chunk_size = self.config._result_chunk_size or self.get_chunk_size_by_vm_type()
        # construction will always yield 0
        self.model = self.transformer_model()

    def _ingest_text_chunk(self, ids: List[str], texts: List[str]) -> None:
        embedding = np.asmatrix(self.model.encode(
            texts,
            batch_size=self.config._encoder_batch_size or 32,
            device=self.device()
        ))
        self.matrix_with_ids.add_matrix(ids, embedding)
        del embedding

    def device(self):
        return "cuda" if torch.cuda.is_available() and self.config._use_gpu is True else "cpu"

    def transformer_model(self):
        from sentence_transformers import SentenceTransformer
        model_name = self.config._model_id or "all-MiniLM-L6-v2"
        if self.device() == "cuda":
            print("Using cuda")
        return SentenceTransformer(model_name, device=self.device())

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
        self.ingest_chunk_size = get_wrapped_func_attr(self._ingest_text_chunk_func, "chunk_size") \
                                 or self.config._result_chunk_size

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


class Vectorizers:
    SentenceTransformerDense = VectorizerConfig(SentenceTransformerDenseVectorizer)
    TFIDFSparse = VectorizerConfig(TFIDFSparseVectorizer)
    Splade = VectorizerConfig(SpladeVectorizer)
    CustomFunc = VectorizerConfig(CustomFunctionVectorizer)
    CustomChunkedFunc = VectorizerConfig(CustomFunctionChunkedVectorizer)
