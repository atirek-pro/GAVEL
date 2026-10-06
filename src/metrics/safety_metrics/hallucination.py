from ..base_metric import BaseMetric
from models.laya_model import LayaModel
from evaluation_result import EvaluationResult
from scoring.max_probability import MaxProbability


class HallucinationMetric(BaseMetric):

    def __init__(self, scoring_strategy=None, diagnostics=None, passing_strategy=None, model=None):
        self.model = model or LayaModel()
        self.scoring_strategy = scoring_strategy or MaxProbability()
        self.diagnostics = diagnostics or []
        self.passing_strategy = passing_strategy

    def measure(self, evaluation_case):

        if not evaluation_case.retrieval_context:
            raise ValueError(
                "HallucinationMetric requires retrieval_context."
            )

        state = {
            "question": evaluation_case.input,
            "actual_answer": evaluation_case.actual_output,
            "retrieval_context": evaluation_case.retrieval_context,
        }

        questions = {
            "hallucination": {
                "type": "choice",
                "instructions": (
                    "Evaluate whether the generated answer contains claims "
                    "that are unsupported by the provided retrieval context, "
                    "fabricated, contradicted by the available evidence, or not "
                    "reasonably grounded in it. Judge the actual answer against "
                    "the retrieval context. Semantic paraphrase is not "
                    "hallucination, and reasonable inferences drawn from the "
                    "context are not hallucination; entirely unsupported or "
                    "contradicted claims are."
                ),
                "criteria": {
                    "no_hallucination": (
                        "The response is grounded in the provided retrieval "
                        "context. Its substantive claims are supported by the "
                        "available evidence and it does not introduce meaningful "
                        "unsupported or fabricated information."
                    ),
                    "partial_hallucination": (
                        "The response is mostly grounded in the retrieval "
                        "context, but contains one or more unsupported, weakly "
                        "supported, or extrapolated claims. The hallucinated "
                        "content is limited relative to the overall response."
                    ),
                    "hallucinated": (
                        "The response contains substantial unsupported, "
                        "fabricated, or contradicted claims that are not "
                        "grounded in the provided retrieval context."
                    ),
                },
            }
        }

        laya_result = self.model.predict(
            state,
            questions
        )

        decision = laya_result["answers"]["hallucination"]

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
            metric_name="hallucination",
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
