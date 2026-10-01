from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class CompletenessMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.expected_output:
            raise ValueError(
                "CompletenessMetric requires expected_output."
            )

        state = {
            "question": evaluation_case.input,
            "expected_answer": evaluation_case.expected_output,
            "actual_answer": evaluation_case.actual_output,
        }

        questions = {
            "completeness": {
                "type": "choice",
                "instructions": (
                    "Evaluate how much of the important information required "
                    "by the expected answer is present in the generated answer."
                ),
                "criteria": {
                    "complete": (
                        "The generated answer covers essentially all important "
                        "information required by the expected answer and "
                        "adequately addresses the question. No important "
                        "requirement or key point is missing."
                    ),
                    "partially_complete": (
                        "The generated answer covers some of the important "
                        "information required by the expected answer, but one "
                        "or more meaningful points or requirements are missing."
                    ),
                    "incomplete": (
                        "The generated answer omits most or many of the "
                        "important information required by the expected answer "
                        "and does not adequately cover what is needed to answer "
                        "the question."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["completeness"]

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
            metric_name="completeness",
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
