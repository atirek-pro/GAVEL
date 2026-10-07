from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class TrajectoryEvaluationMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.trajectory:
            raise ValueError(
                "TrajectoryEvaluationMetric requires trajectory."
            )

        for entry in evaluation_case.trajectory:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Trajectory entries must be dictionaries."
                )
            if "action" not in entry:
                raise ValueError(
                    "Each trajectory entry must contain an 'action' field."
                )

        state = {
            "task": evaluation_case.input,
            "trajectory": evaluation_case.trajectory,
        }

        if evaluation_case.expected_output:
            state["expected_outcome"] = evaluation_case.expected_output

        questions = {
            "trajectory_evaluation": {
                "type": "choice",
                "instructions": (
                    "Given the user's task and the ordered sequence of actions "
                    "the agent took, evaluate whether the overall trajectory was "
                    "appropriate, efficient, and logically aligned with "
                    "accomplishing the task. Judge the sequence as a whole, "
                    "considering unnecessary or repeated actions, logical "
                    "ordering, missing necessary steps, contradictory actions, "
                    "premature termination, and inefficient tool usage. Do not "
                    "reduce this to whether the task was completed, and do not "
                    "independently score each tool call's arguments."
                ),
                "criteria": {
                    "appropriate": (
                        "The agent followed a logical and effective sequence of "
                        "actions. The actions were relevant to the task, "
                        "necessary steps were performed, the ordering was "
                        "sensible, and there were no significant unnecessary, "
                        "contradictory, or harmful actions."
                    ),
                    "partially_appropriate": (
                        "The trajectory generally moved toward accomplishing the "
                        "task but contained some unnecessary actions, "
                        "inefficient steps, minor ordering issues, repetition, or "
                        "missing non-critical actions."
                    ),
                    "inappropriate": (
                        "The trajectory contains major problems such as "
                        "irrelevant actions, incorrect sequencing, repeated "
                        "failed actions, contradictory actions, missing critical "
                        "steps, premature termination, or actions that "
                        "substantially interfere with accomplishing the task."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["trajectory_evaluation"]

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
            metric_name="trajectory_evaluation",
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
                "trajectory": evaluation_case.trajectory,
            },
        )
