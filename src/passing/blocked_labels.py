from .base import BasePassingStrategy


class BlockedLabels(BasePassingStrategy):

    def __init__(self, blocked_labels):
        self.blocked_labels = set(blocked_labels)

    def check(self, decision, score=None, diagnostics=None):
        """
        Pass if Laya's selected label is not in the blocked labels.
        """

        return decision["choice"] not in self.blocked_labels