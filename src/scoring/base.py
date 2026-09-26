from abc import ABC, abstractmethod


class BaseScoringStrategy(ABC):

    @abstractmethod
    def calculate(self, decision):
        """
        Calculate a metric score from a Laya decision.

        Parameters
        ----------
        decision : dict
            A single decision returned by Laya.

        Returns
        -------
        float
            Calculated score.
        """
        pass