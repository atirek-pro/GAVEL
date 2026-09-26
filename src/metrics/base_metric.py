from abc import ABC, abstractmethod


class BaseMetric(ABC):

    @abstractmethod
    def measure(self, evaluation_case):
        """
        Evaluate an EvaluationCase.
        """
        pass