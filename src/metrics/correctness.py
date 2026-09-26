from .base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult


class CorrectnessMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        state = {
            "question": evaluation_case.input,
            "expected_answer": evaluation_case.expected_output,
            "actual_answer": evaluation_case.actual_output,
        }

        questions = {
            "correctness": {
                "type": "choice",
                "instructions": (
                    "Evaluate the correctness of the generated answer "
                    "compared with the expected answer."
                ),
                "criteria": {
                    "fully_correct": (
                        "The generated answer is completely correct."
                    ),
                    "partially_correct": (
                        "The generated answer is partially correct."
                    ),
                    "incorrect": (
                        "The generated answer is incorrect."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["correctness"]

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
            metric_name="correctness",
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