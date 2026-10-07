from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class ToolCorrectnessMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.selected_tool:
            raise ValueError(
                "ToolCorrectnessMetric requires selected_tool."
            )

        if not evaluation_case.available_tools:
            raise ValueError(
                "ToolCorrectnessMetric requires available_tools."
            )

        if not evaluation_case.tool_arguments:
            raise ValueError(
                "ToolCorrectnessMetric requires tool_arguments."
            )

        selected_tool_definition = None
        for tool in evaluation_case.available_tools:
            if isinstance(tool, dict) and tool.get("name") == evaluation_case.selected_tool:
                selected_tool_definition = tool
                break

        if selected_tool_definition is None:
            raise ValueError(
                "Selected tool definition not found in available_tools."
            )

        state = {
            "task": evaluation_case.input,
            "selected_tool": evaluation_case.selected_tool,
            "tool_description": selected_tool_definition["description"],
            "tool_parameters": selected_tool_definition["parameters"],
            "tool_arguments": evaluation_case.tool_arguments,
        }

        questions = {
            "tool_correctness": {
                "type": "choice",
                "instructions": (
                    "Given the user's task and the selected tool with its "
                    "expected parameters, evaluate whether the arguments the "
                    "agent supplied are correct and appropriate for performing "
                    "the task. Judge the argument values semantically against "
                    "the task, not by exact string equality; do not judge "
                    "whether a different tool should have been selected."
                ),
                "criteria": {
                    "correct": (
                        "The selected tool was provided with the correct "
                        "arguments needed to perform the requested task. "
                        "Required values are correct, relevant, and consistent "
                        "with the user's request."
                    ),
                    "partially_correct": (
                        "The tool arguments are partially correct, but one or "
                        "more non-critical values are missing, imprecise, or "
                        "suboptimal. The call captures the main intent but "
                        "requires some correction."
                    ),
                    "incorrect": (
                        "The tool arguments are incorrect, substantially "
                        "incomplete, irrelevant, or contradict the user's "
                        "request, causing the tool call to perform the wrong "
                        "operation or fail to satisfy the task."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["tool_correctness"]

        diagnostic_results = {}
        for diagnostic in self.diagnostics:
            diagnostic_name = diagnostic.__class__.__name__
            diagnostic_results[diagnostic_name] = diagnostic.calculate(decision)

        score = None

        if self.scoring_strategy:
            score = self.scoring_strategy.calculate(decision)

        passed = None

        if self.passing_strategy:
            passed = self.passing_strategy.check(
                decision=decision,
                score=score,
                diagnostics=diagnostic_results
            )

        return EvaluationResult(
            metric_name="tool_correctness",
            score=score,
            passed=passed,
            label=decision["choice"],
            diagnostics=diagnostic_results,
            details={
                "decision": decision,
                "probabilities": decision["probabilities"],
                "confidence": decision["confidence"],
                "answer_confidence": decision["answer_confidence"],
                "action": decision["action"],
                "model": laya_result["model"],
                "usage": laya_result["usage"],
                "routing": laya_result["routing"],
                "selected_tool": evaluation_case.selected_tool,
                "tool_arguments": evaluation_case.tool_arguments,
            },
        )
