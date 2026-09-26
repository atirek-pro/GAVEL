from diagnostics.base import BaseDiagnostic


class Margin(BaseDiagnostic):

    def calculate(self, decision):
        """
        Calculate the margin between the highest and
        second-highest probabilities.

        Margin = top_probability - second_probability
        """

        probabilities = decision["probabilities"]

        if len(probabilities) < 2:
            raise ValueError(
                "Margin requires at least two labels."
            )

        sorted_probabilities = sorted(
            probabilities.values(),
            reverse=True
        )

        return sorted_probabilities[0] - sorted_probabilities[1]