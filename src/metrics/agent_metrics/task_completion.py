from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class TaskCompletionMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.expected_output:
            raise ValueError(
                "TaskCompletionMetric requires expected_output."
            )

        state = {
            "task": evaluation_case.input,
            "expected_outcome": evaluation_case.expected_output,
            "actual_outcome": evaluation_case.actual_output,
        }

        questions = {
            "task_completion": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the agent accomplished the task "
                    "requested by the user, judging the final outcome against "
                    "the expected outcome. Do not judge individual tool calls "
                    "or the agent's trajectory, only whether the requested task "
                    "was actually completed."
                ),
                "criteria": {
                    "completed": (
                        "The agent successfully accomplished the requested task "
                        "and produced an outcome that satisfies the important "
                        "requirements of the expected outcome."
                    ),
                    "partially_completed": (
                        "The agent accomplished some meaningful parts of the "
                        "requested task, but one or more important requirements "
                        "remain incomplete or unsatisfied."
                    ),
                    "not_completed": (
                        "The agent failed to accomplish the requested task or "
                        "failed to produce a meaningful outcome that satisfies "
                        "the task requirements."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["task_completion"]

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
            metric_name="task_completion",
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
