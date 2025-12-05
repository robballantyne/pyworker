import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Set, Union
import inspect

import psutil

log = logging.getLogger(__file__)


class JsonDataException(Exception):
    def __init__(self, json_msg: Dict[str, Any]):
        self.message = json_msg


@dataclass
class AuthData:
    """data used to authenticate requester"""

    cost: Union[str, float, int]  # Can be string or number (autoscaler sends as number)
    endpoint: str
    reqnum: int
    request_idx: int
    signature: str
    url: str

    @classmethod
    def from_json_msg(cls, json_msg: Dict[str, Any]):
        errors = {}
        for param in inspect.signature(cls).parameters:
            if param not in json_msg:
                errors[param] = "missing parameter"
        if errors:
            raise JsonDataException(errors)
        return cls(
            **{
                k: v
                for k, v in json_msg.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class SystemMetrics:
    """General system metrics"""

    model_loading_start: float
    model_loading_time: Union[float, None]
    last_disk_usage: float
    additional_disk_usage: float
    model_is_loaded: bool

    @staticmethod
    def get_disk_usage_GB():
        return psutil.disk_usage("/").used / (2**30)  # want units of GB

    @classmethod
    def empty(cls):
        return cls(
            model_loading_start=time.time(),
            model_loading_time=None,
            last_disk_usage=SystemMetrics.get_disk_usage_GB(),
            additional_disk_usage=0.0,
            model_is_loaded=False,
        )

    def update_disk_usage(self):
        disk_usage = SystemMetrics.get_disk_usage_GB()
        self.additional_disk_usage = disk_usage - self.last_disk_usage
        self.last_disk_usage = disk_usage

    def reset(self, expected: float | None) -> None:
        # autoscaler excepts model_loading_time to be populated only once, when the instance has
        # finished benchmarking and is ready to receive requests. This applies to restarted instances
        # as well: they should send model_loading_time once when they are done loading
        if self.model_loading_time == expected:
            self.model_loading_time = None


@dataclass
class RequestMetrics:
    """Tracks metrics for an active request."""
    request_idx: int
    reqnum: int
    workload: float
    status: str
    success: bool = False

@dataclass
class ModelMetrics:
    """Model specific metrics"""

    # these are reset after being sent to autoscaler
    workload_served: float
    workload_received: float
    workload_cancelled: float
    workload_errored: float
    workload_rejected: float
    # these are not
    workload_pending: float
    error_msg: Optional[str]
    max_throughput: float
    requests_recieved: Set[int] = field(default_factory=set)
    requests_working: dict[int, RequestMetrics] = field(default_factory=dict)
    requests_deleting: list[RequestMetrics] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    @classmethod
    def empty(cls):
        return cls(
            workload_pending=0.0,
            workload_served=0.0,
            workload_cancelled=0.0,
            workload_errored=0.0,
            workload_rejected=0.0,
            workload_received=0.0,
            error_msg=None,
            max_throughput=0.0,
        )
    
    @property
    def workload_processing(self) -> float:
        return max(self.workload_received - self.workload_cancelled, 0.0)

    @property
    def wait_time(self) -> float:
        if (len(self.requests_working) == 0):
            return 0.0
        return sum([request.workload for request in self.requests_working.values()]) / max(self.max_throughput, 0.00001)
    
    @property
    def cur_load(self) -> float:
        return sum([request.workload for request in self.requests_working.values()])

    @property
    def working_request_idxs(self) -> list[int]:
        return [req.request_idx for req in self.requests_working.values()]

    def set_errored(self, error_msg):
        self.reset()
        self.error_msg = error_msg

    def reset(self):
        self.workload_served = 0
        self.workload_received = 0
        self.workload_cancelled = 0
        self.workload_errored = 0
        self.workload_rejected = 0
        self.last_update = time.time()


@dataclass
class AutoScalerData:
    """Data that is reported to autoscaler"""

    id: int
    mtoken: str
    version: str
    loadtime: float
    cur_load: float
    rej_load: float
    new_load: float
    error_msg: str
    max_perf: float
    cur_perf: float
    cur_capacity: float
    max_capacity: float
    num_requests_working: int
    num_requests_recieved: int
    additional_disk_usage: float
    working_request_idxs: list[int]
    url: str
