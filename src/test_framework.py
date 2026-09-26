from evaluation_case import LLMEvaluationCase
from metrics.correctness import CorrectnessMetric

from scoring.max_probability import MaxProbability
from scoring.expected_utility import ExpectedUtility
from scoring.probability_of import ProbabilityOf

from diagnostics.margin import Margin

from passing.score_threshold import ScoreThreshold
from passing.label_match import LabelMatch
from passing.probability_threshold import ProbabilityThreshold
from passing.allowed_labels import AllowedLabels
from passing.blocked_labels import BlockedLabels

case = LLMEvaluationCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris"
)


metric = CorrectnessMetric(
    scoring_strategy=MaxProbability(),
    diagnostics=[
        Margin()
    ],
    passing_strategy=BlockedLabels(
        blocked_labels=[
            "fully_correct"
        ]
    )
)

result = metric.measure(case)


print("\n========== EVALUATION RESULT ==========")

print("Metric:", result.metric_name)
print("Label:", result.label)
print("Score:", result.score)
print("Diagnois:", result.diagnostics)
print("Passed:", result.passed)

print("\nDetails:")
print(result.details)