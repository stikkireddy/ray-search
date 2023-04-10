import json
from dataclasses import dataclass
from enum import Enum
from json import JSONEncoder
from typing import Iterator, List, Dict, Union, Optional, Any

import numpy as np
import pandas as pd
import ray
from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster


@dataclass
class Window:
    start: int
    end: int


def window_iter(total_window_size: int, chunk_size: int) -> Iterator[Window]:
    # looping till length l
    for idx, i in enumerate(range(0, total_window_size, chunk_size)):
        yield Window(i, min(i + chunk_size, total_window_size))


class VmType(Enum):
    LOW_MEMORY = 1
    MED_MEMORY = 2
    HIGH_MEMORY = 3


class Memory:

    def __init__(self, bytes_: int):
        self._bytes: int = bytes_

    def __repr__(self):
        return f"<Memory: {self._bytes} bytes>"

    @property
    def bytes(self) -> int:
        return self._bytes

    @classmethod
    def in_gb(cls, num_gb: int):
        return cls(num_gb * 1024 ** 3)

    @classmethod
    def in_mb(cls, num_mb: int):
        return cls(num_mb * 1024 ** 2)

    @classmethod
    def in_kb(cls, num_kb: int):
        return cls(num_kb * 1024 ** 1)


class NumpyValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)


@dataclass
class SearchResult:
    __slots__ = ('input_id', 'search_results')

    input_id: str
    search_results: List[Dict[str, Union[str, float]]]

    def dict(self):
        return {
            "input_id": self.input_id,
            "search_results": self.search_results
        }

    def unnested_list(self, content_map: Optional[Dict[str, str]] = None):
        lookup = content_map or {}
        return [
            {
                "input_id": self.input_id,
                "search_result_id": res["search_result_id"],
                "score": res["score"],
                "raw_content": lookup.get(res["search_result_id"]),
            }
            for res in self.search_results
        ]

    @staticmethod
    def to_list(results: List['SearchResult'],
                unnest: bool = False,
                content_map: Optional[Dict[str, str]] = None
                ) -> List:
        if unnest is False:
            return [res.dict() for res in results]
        return [un_nested for res in results for un_nested in res.unnested_list(content_map)]

    @staticmethod
    def to_json(results: List['SearchResult'],
                unnest: bool = False,
                indent=None,
                content_map: Optional[Dict[str, str]] = None) -> str:
        return json.dumps({"results": SearchResult.to_list(results, unnest, content_map)},
                          indent=indent, cls=NumpyValuesEncoder)

    @staticmethod
    def to_df(results: List['SearchResult'],
              unnest: bool = False,
              content_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        return pd.DataFrame.from_records(SearchResult.to_list(results, unnest, content_map))


class RayCluster:

    def __init__(self, num_worker_nodes: int = 2,
                 num_cpus_per_node: int = 4,
                 runtime_env: Dict[str, Any] = None,
                 num_gpus_per_node: Optional[int] = None):
        self.num_worker_nodes = num_worker_nodes
        self.num_cpus_per_node = num_cpus_per_node
        self.runtime_env = runtime_env
        self.num_gpus_per_node = num_gpus_per_node

    def max_parallel_workers(self, num_cpus_per_worker: int = 1):
        return int((self.num_worker_nodes * self.num_cpus_per_node) / num_cpus_per_worker)

    def start(self):
        setup_ray_cluster(
            num_worker_nodes=self.num_worker_nodes,
            num_cpus_per_node=self.num_cpus_per_node,
            num_gpus_per_node=self.num_gpus_per_node
        )
        if self.runtime_env is not None:
            ray.init(runtime_env=self.runtime_env)
        else:
            ray.init()

    def stop(self):
        shutdown_ray_cluster()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
