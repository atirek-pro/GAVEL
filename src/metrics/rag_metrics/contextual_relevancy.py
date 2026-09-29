from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class ContextualRelevancyMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.retrieval_context:
            raise ValueError(
                "ContextualRelevancyMetric requires retrieval_context."
            )

        state = {
            "question": evaluation_case.input,
            "retrieval_context": evaluation_case.retrieval_context,
        }

        questions = {
            "contextual_relevancy": {
                "type": "choice",
                "instructions": (
                    "Evaluate the relevance of the retrieved context "
                    "to the user's question."
                ),
                "criteria": {
                    "highly_relevant": (
                        "The retrieved context is directly relevant to the "
                        "user's question and contains information that is "
                        "useful for answering it."
                    ),
                    "partially_relevant": (
                        "The retrieved context contains some information "
                        "relevant to the user's question, but also contains "
                        "irrelevant information, or does not sufficiently "
                        "cover the question."
                    ),
                    "irrelevant": (
                        "The retrieved context does not meaningfully help "
                        "answer the user's question and is unrelated or "
                        "mostly unrelated."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["contextual_relevancy"]

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
            metric_name="contextual_relevancy",
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
