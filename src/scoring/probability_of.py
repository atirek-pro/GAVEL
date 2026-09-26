from .base import BaseScoringStrategy


class ProbabilityOf(BaseScoringStrategy):

    def __init__(self, label):
        self.label = label

    def calculate(self, decision):
        """
        Return the probability assigned to a specific label.
        """

        probabilities = decision["probabilities"]

        if self.label not in probabilities:
            raise ValueError(
                f"Label '{self.label}' not found in decision probabilities."
            )

        return probabilities[self.label]