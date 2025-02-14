from abc import ABC,abstractmethod
from lib.utils import Constant


class BizAlgo(ABC):
    def __init__(self) -> None:
        self.const=Constant.get()
    @abstractmethod
    def _preprocess(self)->None:
        ...
    @abstractmethod
    def get_result(self):
        ...
    @abstractmethod
    def _postprocess(self)->None:
        ...
