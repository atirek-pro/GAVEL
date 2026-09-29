from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class ContextualRecallMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.expected_output:
            raise ValueError(
                "ContextualRecallMetric requires expected_output."
            )

        if not evaluation_case.retrieval_context:
            raise ValueError(
                "ContextualRecallMetric requires retrieval_context."
            )

        state = {
            "question": evaluation_case.input,
            "expected_answer": evaluation_case.expected_output,
            "retrieval_context": evaluation_case.retrieval_context,
        }

        questions = {
            "contextual_recall": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the retrieved context contains the "
                    "information needed to support the expected answer."
                ),
                "criteria": {
                    "high_recall": (
                        "The retrieved context contains the important "
                        "information needed to support essentially all of the "
                        "expected answer. The necessary facts or evidence are "
                        "present in the retrieved context."
                    ),
                    "partial_recall": (
                        "The retrieved context contains some of the information "
                        "needed to support the expected answer, but one or more "
                        "important facts or pieces of information are missing."
                    ),
                    "low_recall": (
                        "The retrieved context does not contain the information "
                        "needed to support the expected answer, or contains very "
                        "little of the required information."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["contextual_recall"]

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
            metric_name="contextual_recall",
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
