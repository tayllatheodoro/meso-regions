import abc
from abc import ABC


class AbstractMethod(ABC):
    def __init__(self, **kwargs):
        self.thread_safe = True

    @abc.abstractmethod
    def apply(self, patient, **kwargs):
        pass

    def result(self):
        return None
