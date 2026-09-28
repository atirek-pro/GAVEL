from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class RelevanceMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        state = {
            "question": evaluation_case.input,
            "actual_answer": evaluation_case.actual_output,
        }

        questions = {
            "relevance": {
                "type": "choice",
                "instructions": (
                    "Evaluate the relevance of the generated answer "
                    "to the user's question."
                ),
                "criteria": {
                    "fully_relevant": (
                        "The generated answer directly addresses the user's "
                        "question and provides information relevant to what "
                        "was asked."
                    ),
                    "partially_relevant": (
                        "The generated answer addresses some aspect of the "
                        "user's question but is incomplete, indirect, or "
                        "contains substantial irrelevant information."
                    ),
                    "irrelevant": (
                        "The generated answer does not meaningfully address "
                        "the user's question."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["relevance"]

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
            metric_name="relevance",
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
