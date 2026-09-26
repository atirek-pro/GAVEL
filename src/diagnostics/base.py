from abc import ABC, abstractmethod


class BaseDiagnostic(ABC):

    @abstractmethod
    def calculate(self, decision):
        """
        Calculate a diagnostic value from a Laya decision.

        Parameters
        ----------
        decision : dict
            A single decision returned by Laya.

        Returns
        -------
        float
            Diagnostic value.
        """
        pass