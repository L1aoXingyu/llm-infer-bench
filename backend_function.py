from query_vllm import query_model_vllm
from query_lightllm import query_model_lightllm
from query_tgi import query_model_tgi
from enum import Enum


class Backend(str, Enum):
    VLLM = "vllm"
    LIGHTLLM = "lightllm"
    TGI = "tgi"


class BackendFunctionRegistry:
    registry = {}

    @classmethod
    def register(cls, backend, func):
        cls.registry[backend] = func

    @classmethod
    def get_function(cls, backend):
        return cls.registry.get(backend)


BackendFunctionRegistry.register(Backend.VLLM, query_model_vllm)
BackendFunctionRegistry.register(Backend.LIGHTLLM, query_model_lightllm)
BackendFunctionRegistry.register(Backend.TGI, query_model_tgi)
