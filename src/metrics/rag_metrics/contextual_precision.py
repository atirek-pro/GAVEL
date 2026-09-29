from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class ContextualPrecisionMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.expected_output:
            raise ValueError(
                "ContextualPrecisionMetric requires expected_output."
            )

        if not evaluation_case.retrieval_context:
            raise ValueError(
                "ContextualPrecisionMetric requires retrieval_context."
            )

        state = {
            "question": evaluation_case.input,
            "expected_answer": evaluation_case.expected_output,
            "retrieval_context": evaluation_case.retrieval_context,
        }

        questions = {
            "contextual_precision": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the retrieved context prioritizes the "
                    "information that is relevant to the expected answer, "
                    "judging the ordering of the retrieved chunks. Consider "
                    "which chunks support the expected answer, the order in "
                    "which they appear, and whether relevant chunks are placed "
                    "before irrelevant chunks."
                ),
                "criteria": {
                    "high_precision": (
                        "The retrieved context places the most relevant "
                        "information needed to support the expected answer early "
                        "in the retrieved results, with little or no irrelevant "
                        "information preceding relevant information."
                    ),
                    "partial_precision": (
                        "The retrieved context contains relevant information, "
                        "but some irrelevant or less useful chunks are placed "
                        "before relevant information, reducing retrieval "
                        "precision."
                    ),
                    "low_precision": (
                        "Relevant information is poorly prioritized, with "
                        "substantial irrelevant information appearing before or "
                        "among the useful retrieved information, or relevant "
                        "information is largely buried in the retrieval results."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["contextual_precision"]

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
            metric_name="contextual_precision",
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
