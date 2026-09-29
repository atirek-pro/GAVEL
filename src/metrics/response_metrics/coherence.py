from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class CoherenceMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        state = {
            "question": evaluation_case.input,
            "actual_answer": evaluation_case.actual_output,
        }

        questions = {
            "coherence": {
                "type": "choice",
                "instructions": (
                    "Evaluate how logically organized, internally consistent, "
                    "and easy to follow the generated answer is."
                ),
                "criteria": {
                    "coherent": (
                        "The response is logically organized, internally "
                        "consistent, and easy to follow. Ideas connect "
                        "naturally and there are no meaningful logical or "
                        "structural problems."
                    ),
                    "partially_coherent": (
                        "The response is understandable overall, but contains "
                        "some confusing transitions, minor logical jumps, "
                        "inconsistencies, or structural issues."
                    ),
                    "incoherent": (
                        "The response is difficult to follow because ideas are "
                        "poorly connected, contradictory, disorganized, or "
                        "logically unclear."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["coherence"]

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
            metric_name="coherence",
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
