from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class FaithfulnessMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.retrieval_context:
            raise ValueError(
                "FaithfulnessMetric requires retrieval_context."
            )

        state = {
            "question": evaluation_case.input,
            "actual_answer": evaluation_case.actual_output,
            "retrieval_context": evaluation_case.retrieval_context,
        }

        questions = {
            "faithfulness": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the claims in the generated answer "
                    "are supported by the retrieved context."
                ),
                "criteria": {
                    "fully_faithful": (
                        "The claims in the generated answer are fully supported "
                        "by the retrieved context. The answer does not introduce "
                        "unsupported factual claims."
                    ),
                    "partially_faithful": (
                        "The generated answer is partially supported by the "
                        "retrieved context, but contains one or more claims that "
                        "are unsupported, incomplete, or only partially supported."
                    ),
                    "unfaithful": (
                        "The generated answer contains claims that are not "
                        "supported by the retrieved context or substantially "
                        "contradicts the retrieved context."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["faithfulness"]

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
            metric_name="faithfulness",
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
