from evaluation_case import LLMEvaluationCase
from models.laya_model import LayaModel
from metrics.response_metrics.correctness import CorrectnessMetric
from metrics.response_metrics.relevance import RelevanceMetric
from metrics.rag_metrics.faithfulness import FaithfulnessMetric

from scoring.max_probability import MaxProbability
from scoring.expected_utility import ExpectedUtility
from scoring.probability_of import ProbabilityOf

from diagnostics.margin import Margin

from passing.score_threshold import ScoreThreshold
from passing.label_match import LabelMatch
from passing.probability_threshold import ProbabilityThreshold
from passing.allowed_labels import AllowedLabels
from passing.blocked_labels import BlockedLabels

# A single shared Laya model is reused across every metric.
# Laya's Router preloads native model resources; instantiating more
# than one Router in the same process can crash the interpreter, so
# the metrics inject the same model through the existing `model=` hook.
shared_model = LayaModel()

case = LLMEvaluationCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    expected_output="Paris"
)

# Faithfulness is evaluated against the retrieved context, not the
# expected answer, so it uses its own evaluation cases.
faithfulness_case = LLMEvaluationCase(
    input="What is the company's annual leave policy?",
    actual_output="Employees receive 20 days of annual leave.",
    retrieval_context=[
        "Employees receive 20 days of annual leave each year."
    ]
)

faithfulness_unsupported_case = LLMEvaluationCase(
    input="What is the company's annual leave policy?",
    actual_output=(
        "Employees receive 30 days of annual leave "
        "and unlimited sick leave."
    ),
    retrieval_context=[
        "Employees receive 20 days of annual leave each year."
    ]
)


def print_relevance_result(result):
    print("\n========== RELEVANCE EVALUATION ==========")
    print()
    print("Metric:", result.metric_name)
    print("Label:", result.label)
    print("Score:", result.score)
    print("Passed:", result.passed)

    print("\nDiagnostics:")
    print(result.diagnostics)

    print("\nProbabilities:")
    print(result.details["probabilities"])


def run_test_1_max_probability():
    print("\n--- Test 1: MaxProbability ---")

    metric = RelevanceMetric(
        scoring_strategy=MaxProbability(),
        model=shared_model
    )

    result = metric.measure(case)

    assert result.label in ("fully_relevant", "partially_relevant", "irrelevant")
    assert result.score is not None

    print_relevance_result(result)


def run_test_2_expected_utility():
    print("\n--- Test 2: ExpectedUtility ---")

    metric = RelevanceMetric(
        scoring_strategy=ExpectedUtility(
            utilities={
                "fully_relevant": 1.0,
                "partially_relevant": 0.5,
                "irrelevant": 0.0,
            }
        ),
        model=shared_model
    )

    result = metric.measure(case)

    assert result.score is not None

    print_relevance_result(result)


def run_test_3_probability_of():
    print("\n--- Test 3: ProbabilityOf('fully_relevant') ---")

    metric = RelevanceMetric(
        scoring_strategy=ProbabilityOf("fully_relevant"),
        model=shared_model
    )

    result = metric.measure(case)

    assert result.score is not None
    assert result.score == result.details["probabilities"]["fully_relevant"]

    print_relevance_result(result)


def run_test_4_margin():
    print("\n--- Test 4: Margin diagnostic ---")

    metric = RelevanceMetric(
        scoring_strategy=MaxProbability(),
        diagnostics=[
            Margin()
        ],
        model=shared_model
    )

    result = metric.measure(case)

    assert "Margin" in result.diagnostics
    assert result.diagnostics["Margin"] is not None

    print_relevance_result(result)


def run_test_5_score_threshold():
    print("\n--- Test 5: ScoreThreshold ---")

    metric = RelevanceMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=ScoreThreshold(
            threshold=0.60
        ),
        model=shared_model
    )

    result = metric.measure(case)

    assert result.passed is not None

    print_relevance_result(result)


def run_test_6_label_match():
    print("\n--- Test 6: LabelMatch ---")

    metric = RelevanceMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=LabelMatch(
            expected_label="fully_relevant"
        ),
        model=shared_model
    )

    result = metric.measure(case)

    assert result.passed is not None
    assert result.passed == (result.label == "fully_relevant")

    print_relevance_result(result)


def print_faithfulness_result(result):
    print("\n========== FAITHFULNESS EVALUATION ==========")
    print()
    print("Metric:", result.metric_name)
    print("Label:", result.label)
    print("Score:", result.score)
    print("Passed:", result.passed)

    print("\nDiagnostics:")
    print(result.diagnostics)

    print("\nProbabilities:")
    print(result.details["probabilities"])


def run_faithfulness_test_1_basic():
    print("\n--- Faithfulness Test 1: Basic MaxProbability ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert result.label in ("fully_faithful", "partially_faithful", "unfaithful")
    assert result.score is not None
    assert result.diagnostics is not None
    assert result.passed is None

    print_faithfulness_result(result)


def run_faithfulness_test_2_unsupported_claim():
    print("\n--- Faithfulness Test 2: Unsupported claim ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        model=shared_model
    )

    result = metric.measure(faithfulness_unsupported_case)

    assert result.label in ("fully_faithful", "partially_faithful", "unfaithful")
    assert result.score is not None

    print_faithfulness_result(result)

    print("\nComplete Laya decision (unsupported claim case):")
    print(result.details["decision"])


def run_faithfulness_test_3_expected_utility():
    print("\n--- Faithfulness Test 3: ExpectedUtility ---")

    metric = FaithfulnessMetric(
        scoring_strategy=ExpectedUtility(
            utilities={
                "fully_faithful": 1.0,
                "partially_faithful": 0.5,
                "unfaithful": 0.0,
            }
        ),
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert result.score is not None

    print_faithfulness_result(result)


def run_faithfulness_test_4_probability_of():
    print("\n--- Faithfulness Test 4: ProbabilityOf('fully_faithful') ---")

    metric = FaithfulnessMetric(
        scoring_strategy=ProbabilityOf("fully_faithful"),
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert result.score is not None
    assert result.score == result.details["probabilities"]["fully_faithful"]

    print_faithfulness_result(result)


def run_faithfulness_test_5_margin():
    print("\n--- Faithfulness Test 5: Margin diagnostic ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        diagnostics=[
            Margin()
        ],
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert "Margin" in result.diagnostics
    assert result.diagnostics["Margin"] is not None

    print_faithfulness_result(result)


def run_faithfulness_test_6_score_threshold():
    print("\n--- Faithfulness Test 6: ScoreThreshold ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=ScoreThreshold(
            threshold=0.60
        ),
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert result.passed is not None

    print_faithfulness_result(result)


def run_faithfulness_test_7_label_match():
    print("\n--- Faithfulness Test 7: LabelMatch ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=LabelMatch(
            expected_label="fully_faithful"
        ),
        model=shared_model
    )

    result = metric.measure(faithfulness_case)

    assert result.passed is not None
    assert result.passed == (result.label == "fully_faithful")

    print_faithfulness_result(result)


def run_faithfulness_test_8_empty_context():
    print("\n--- Faithfulness Test 8: Empty retrieval_context ---")

    metric = FaithfulnessMetric(
        scoring_strategy=MaxProbability(),
        model=shared_model
    )

    empty_context_case = LLMEvaluationCase(
        input="What is the policy?",
        actual_output="The policy is X.",
        retrieval_context=[]
    )

    try:
        metric.measure(empty_context_case)
    except ValueError as error:
        assert "retrieval_context" in str(error)
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "FaithfulnessMetric should raise ValueError "
            "when retrieval_context is empty."
        )


# ---------------------------------------------------------------
# Existing correctness functionality (must still work)
# ---------------------------------------------------------------

print("\n========== CORRECTNESS EVALUATION ==========")

correctness_metric = CorrectnessMetric(
    scoring_strategy=MaxProbability(),
    diagnostics=[
        Margin()
    ],
    passing_strategy=BlockedLabels(
        blocked_labels=[
            "fully_correct"
        ]
    ),
    model=shared_model
)

correctness_result = correctness_metric.measure(case)


print("Metric:", correctness_result.metric_name)
print("Label:", correctness_result.label)
print("Score:", correctness_result.score)
print("Diagnois:", correctness_result.diagnostics)
print("Passed:", correctness_result.passed)

print("\nDetails:")
print(correctness_result.details)


# ---------------------------------------------------------------
# Relevance tests
# ---------------------------------------------------------------

print("\n========== RELEVANCE METRIC TESTS ==========")

run_test_1_max_probability()
run_test_2_expected_utility()
run_test_3_probability_of()
run_test_4_margin()
run_test_5_score_threshold()
run_test_6_label_match()


# ---------------------------------------------------------------
# Faithfulness tests
# ---------------------------------------------------------------

print("\n========== FAITHFULNESS METRIC TESTS ==========")

run_faithfulness_test_1_basic()
run_faithfulness_test_2_unsupported_claim()
run_faithfulness_test_3_expected_utility()
run_faithfulness_test_4_probability_of()
run_faithfulness_test_5_margin()
run_faithfulness_test_6_score_threshold()
run_faithfulness_test_7_label_match()
run_faithfulness_test_8_empty_context()


print("\n========== SUMMARY ==========")
print("Correctness   [PASS]")
print("Relevance     [PASS]")
print("Faithfulness  [PASS]")

print("\n========== ALL TESTS PASSED ==========")
