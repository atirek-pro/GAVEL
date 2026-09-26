from abc import ABC, abstractmethod


class BasePassingStrategy(ABC):

    @abstractmethod
    def check(self, decision, score=None, diagnostics=None):
        """
        Determine whether an evaluation passes.

        Parameters
        ----------
        decision : dict
            A single decision returned by Laya.

        score : float | None
            Score produced by a scoring strategy.

        diagnostics : dict | None
            Diagnostic values produced during evaluation.

        Returns
        -------
        bool
            Whether the evaluation passed.
        """
        pass