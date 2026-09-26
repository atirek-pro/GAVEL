from .base import BasePassingStrategy


class ScoreThreshold(BasePassingStrategy):

    def __init__(self, threshold):
        self.threshold = threshold

    def check(self, decision, score=None, diagnostics=None):
        """
        Pass if the calculated score is greater than
        or equal to the configured threshold.
        """

        if score is None:
            raise ValueError(
                "ScoreThreshold requires a score."
            )

        return score >= self.threshold