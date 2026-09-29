from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class ConcisenessMetric(BaseMetric):

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
            "conciseness": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the generated answer communicates the "
                    "needed information without unnecessary repetition, "
                    "verbosity, or irrelevant detail. Judge unnecessary "
                    "content, not response length: a longer answer can still "
                    "be concise when all of its detail is useful."
                ),
                "criteria": {
                    "concise": (
                        "The response communicates the necessary information "
                        "clearly without unnecessary repetition, verbosity, or "
                        "irrelevant detail."
                    ),
                    "partially_concise": (
                        "The response is generally useful but contains some "
                        "unnecessary repetition, verbosity, or extra detail."
                    ),
                    "verbose": (
                        "The response contains substantial unnecessary "
                        "information, repetition, or excessive explanation "
                        "relative to what the question requires."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["conciseness"]

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
            metric_name="conciseness",
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
