from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class ToolSelectionMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.available_tools:
            raise ValueError(
                "ToolSelectionMetric requires available_tools."
            )

        if not evaluation_case.selected_tool:
            raise ValueError(
                "ToolSelectionMetric requires selected_tool."
            )

        state = {
            "task": evaluation_case.input,
            "available_tools": evaluation_case.available_tools,
            "selected_tool": evaluation_case.selected_tool,
        }

        questions = {
            "tool_selection": {
                "type": "choice",
                "instructions": (
                    "Given the user's task and the tools available to the "
                    "agent, evaluate whether the selected tool is appropriate "
                    "for accomplishing the task. Judge the semantic suitability "
                    "of the selected tool for the task relative to the other "
                    "available tools."
                ),
                "criteria": {
                    "appropriate": (
                        "The selected tool is the most appropriate or a clearly "
                        "suitable tool for accomplishing the user's task given "
                        "the available tools."
                    ),
                    "partially_appropriate": (
                        "The selected tool could accomplish part of the task or "
                        "is usable, but another available tool would have been "
                        "more appropriate, efficient, or better suited to the "
                        "task."
                    ),
                    "inappropriate": (
                        "The selected tool is not suitable for accomplishing "
                        "the user's task and does not meaningfully address the "
                        "requirements of the task."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["tool_selection"]

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
            metric_name="tool_selection",
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
            },
        )
