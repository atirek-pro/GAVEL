class LLMEvaluationCase:

    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: str | None = None,
        retrieval_context: list[str] | None = None,
        available_tools: list[str] | None = None,
        selected_tool: str | None = None,
        tool_arguments: dict | None = None,
        trajectory: list[dict] | None = None,
    ):
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.retrieval_context = retrieval_context or []
        self.available_tools = available_tools or []
        self.selected_tool = selected_tool
        self.tool_arguments = tool_arguments or {}
        self.trajectory = trajectory or []