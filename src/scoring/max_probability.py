from .base import BaseScoringStrategy


class MaxProbability(BaseScoringStrategy):

    def calculate(self, decision):
        """
        Return the probability of Laya's selected choice.

        Example:
        {
            "choice": "fully_correct",
            "probabilities": {
                "fully_correct": 0.6707,
                "partially_correct": 0.2202,
                "incorrect": 0.1091
            }
        }

        Returns
        -------
        float
            Highest probability among the available choices.
        """

        probabilities = decision["probabilities"]

        return max(probabilities.values())