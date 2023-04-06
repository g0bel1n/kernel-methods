from abc import ABC, abstractmethod
from typing import Any, Union


class SubKernel(ABC):
    
        def __init__(self):
            pass
    
        @abstractmethod
        def _kernel(self, sub1, sub2):
            pass
    
        def __call__(self, sub1, sub2) -> Union[float, int]:
            return self._kernel(sub1, sub2)

        def __repr__(self):
            return f"{self.__class__.__name__}()"



class EdgeKernel(SubKernel):
    pass

class NodeKernel(SubKernel):
    pass