from laya import Router


class LayaModel:

    def __init__(self):
        self.router = Router(preload=True)

    def predict(self, state, questions):
        """
        Run Laya inference.

        Parameters
        ----------
        state : dict
            Information that Laya should evaluate.

        questions : dict
            Typed decision questions for Laya.

        Returns
        -------
        dict
            Raw Laya prediction.
        """

        return self.router.predict(
            state,
            questions
        )