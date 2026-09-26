from .base import BaseScoringStrategy


class ExpectedUtility(BaseScoringStrategy):

    def __init__(self, utilities):
        self.utilities = utilities

    def calculate(self, decision):
        """
        Calculate the expected utility of a Laya decision.

        Expected Utility =
            sum(
                probability(label) * utility(label)
                for each label
            )
        """

        probabilities = decision["probabilities"]

        score = 0.0

        for label, probability in probabilities.items():
            utility = self.utilities.get(label, 0.0)
            score += probability * utility

        return score