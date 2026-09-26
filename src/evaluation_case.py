class LLMEvaluationCase:

    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: str | None = None,
        retrieval_context: list[str] | None = None,
    ):
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.retrieval_context = retrieval_context or []