from .base import BasePassingStrategy


class AllowedLabels(BasePassingStrategy):

    def __init__(self, allowed_labels):
        self.allowed_labels = set(allowed_labels)

    def check(self, decision, score=None, diagnostics=None):
        """
        Pass if Laya's selected label is one of the allowed labels.
        """

        return decision["choice"] in self.allowed_labels