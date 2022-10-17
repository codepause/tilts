from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
