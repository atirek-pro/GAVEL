from .base import BasePassingStrategy


class ProbabilityThreshold(BasePassingStrategy):

    def __init__(self, label, threshold):
        self.label = label
        self.threshold = threshold

    def check(self, decision, score=None, diagnostics=None):
        """
        Pass if the probability of the specified label
        is greater than or equal to the configured threshold.
        """

        probabilities = decision["probabilities"]

        if self.label not in probabilities:
            raise ValueError(
                f"Label '{self.label}' not found in decision probabilities."
            )

        return probabilities[self.label] >= self.threshold