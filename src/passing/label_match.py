from .base import BasePassingStrategy


class LabelMatch(BasePassingStrategy):

    def __init__(self, expected_label):
        self.expected_label = expected_label

    def check(self, decision, score=None, diagnostics=None):
        """
        Pass if Laya's selected label matches the expected label.
        """

        return decision["choice"] == self.expected_label