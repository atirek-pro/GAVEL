class EvaluationResult:

    def __init__(
        self,
        metric_name: str,
        score: float | None = None,
        passed: bool | None = None,
        label: str | None = None,
        diagnostics: dict | None = None,
        details: dict | None = None,
    ):
        self.metric_name = metric_name
        self.score = score
        self.passed = passed
        self.label = label
        self.diagnostics = diagnostics or {}
        self.details = details or {}