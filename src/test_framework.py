from evaluation_case import LLMEvaluationCase
from models.laya_model import LayaModel
from metrics.response_metrics.correctness import CorrectnessMetric
from metrics.response_metrics.relevance import RelevanceMetric
from metrics.rag_metrics.faithfulness import FaithfulnessMetric
from metrics.rag_metrics.contextual_relevancy import ContextualRelevancyMetric
from metrics.rag_metrics.contextual_recall import ContextualRecallMetric
from metrics.rag_metrics.contextual_precision import ContextualPrecisionMetric

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

# Contextual Relevancy evaluates the retrieved context against the
# user's question, so it uses its own evaluation cases.
contextual_relevancy_case = LLMEvaluationCase(
    input="What is the refund policy?",
    actual_output="Refunds are available within 30 days.",
    retrieval_context=[
        "Customers can request a refund within 30 days of purchase.",
        "Refund requests must be submitted through the customer portal."
    ]
)

contextual_relevancy_irrelevant_case = LLMEvaluationCase(
    input="What is the refund policy?",
    actual_output="Refunds are available within 30 days.",
    retrieval_context=[
        "The company was founded in 2012.",
        "The CEO joined the company in 2018.",
        "The company currently has 500 employees."
    ]
)

# Contextual Recall evaluates whether the retrieved context contains the
# information needed to support the expected answer, so it uses its own
# evaluation cases with an expected_output.
contextual_recall_case = LLMEvaluationCase(
    input="What are the company's refund conditions?",
    actual_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    expected_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    retrieval_context=[
        "Customers can request a refund within 30 days.",
        "A valid refund request requires the original receipt."
    ]
)

contextual_recall_partial_case = LLMEvaluationCase(
    input="What are the company's refund conditions?",
    actual_output="Customers can request a refund within 30 days.",
    expected_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    retrieval_context=[
        "Customers can request a refund within 30 days.",
        "Refund requests are submitted through the customer portal."
    ]
)

# Contextual Precision evaluates the ordering of the retrieved chunks
# against the expected answer (relevant chunks should rank first).
contextual_precision_case = LLMEvaluationCase(
    input="What are the company's refund conditions?",
    actual_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    expected_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    retrieval_context=[
        "Customers can request a refund within 30 days.",
        "A valid refund request requires the original receipt.",
        "Refund requests are submitted through the customer portal.",
        "The company was founded in 2012."
    ]
)

contextual_precision_buried_case = LLMEvaluationCase(
    input="What are the company's refund conditions?",
    actual_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    expected_output=(
        "Customers can request a refund within 30 days and must "
        "provide the original receipt."
    ),
    retrieval_context=[
        "The company was founded in 2012.",
        "The company currently has 500 employees.",
        "The CEO joined the company in 2018.",
        "Customers can request a refund within 30 days.",
        "A valid refund request requires the original receipt."
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


def print_contextual_relevancy_result(result):
    print("\n========== CONTEXTUAL RELEVANCY EVALUATION ==========")
    print()
    print("Metric:", result.metric_name)
    print("Label:", result.label)
    print("Score:", result.score)
    print("Passed:", result.passed)

    print("\nDiagnostics:")
    print(result.diagnostics)

    print("\nProbabilities:")
    print(result.details["probabilities"])


def test_contextual_relevancy():
    print("\n========== CONTEXTUAL RELEVANCY METRIC TESTS ==========")

    # Test 1 - Basic Contextual Relevancy
    print("\n--- Contextual Relevancy Test 1: Basic ---")

    metric = ContextualRelevancyMetric(model=shared_model)

    result = metric.measure(contextual_relevancy_case)

    assert result.metric_name == "contextual_relevancy"
    assert result.label in ("highly_relevant", "partially_relevant", "irrelevant")
    assert result.details["probabilities"] is not None
    assert set(result.details["probabilities"]) == {
        "highly_relevant",
        "partially_relevant",
        "irrelevant",
    }

    print_contextual_relevancy_result(result)

    # Test 2 - Irrelevant Context
    print("\n--- Contextual Relevancy Test 2: Irrelevant context ---")

    metric = ContextualRelevancyMetric(model=shared_model)

    result = metric.measure(contextual_relevancy_irrelevant_case)

    assert result.label in ("highly_relevant", "partially_relevant", "irrelevant")

    print_contextual_relevancy_result(result)

    # Test 3 - ExpectedUtility
    print("\n--- Contextual Relevancy Test 3: ExpectedUtility ---")

    metric = ContextualRelevancyMetric(
        scoring_strategy=ExpectedUtility(
            utilities={
                "highly_relevant": 1.0,
                "partially_relevant": 0.5,
                "irrelevant": 0.0,
            }
        ),
        model=shared_model
    )

    result = metric.measure(contextual_relevancy_case)

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    print_contextual_relevancy_result(result)

    # Test 4 - ProbabilityOf
    print("\n--- Contextual Relevancy Test 4: ProbabilityOf('highly_relevant') ---")

    metric = ContextualRelevancyMetric(
        scoring_strategy=ProbabilityOf("highly_relevant"),
        model=shared_model
    )

    result = metric.measure(contextual_relevancy_case)

    assert result.score == result.details["probabilities"]["highly_relevant"]

    print_contextual_relevancy_result(result)

    # Test 5 - Margin diagnostic
    print("\n--- Contextual Relevancy Test 5: Margin diagnostic ---")

    metric = ContextualRelevancyMetric(
        scoring_strategy=MaxProbability(),
        diagnostics=[
            Margin()
        ],
        model=shared_model
    )

    result = metric.measure(contextual_relevancy_case)

    assert "Margin" in result.diagnostics
    assert result.diagnostics["Margin"] >= 0

    print_contextual_relevancy_result(result)

    # Test 6 - ScoreThreshold
    print("\n--- Contextual Relevancy Test 6: ScoreThreshold ---")

    metric = ContextualRelevancyMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=ScoreThreshold(
            threshold=0.5
        ),
        model=shared_model
    )

    result = metric.measure(contextual_relevancy_case)

    assert result.passed in (True, False)

    print_contextual_relevancy_result(result)

    # Test 7 - LabelMatch
    print("\n--- Contextual Relevancy Test 7: LabelMatch ---")

    metric = ContextualRelevancyMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=LabelMatch(
            expected_label="highly_relevant"
        ),
        model=shared_model
    )

    result = metric.measure(contextual_relevancy_case)

    assert result.passed in (True, False)

    print_contextual_relevancy_result(result)

    # Test 8 - Empty retrieval context
    print("\n--- Contextual Relevancy Test 8: Empty retrieval_context ---")

    metric = ContextualRelevancyMetric(model=shared_model)

    empty_context_case = LLMEvaluationCase(
        input="What is the refund policy?",
        actual_output="Refunds are available within 30 days.",
        retrieval_context=[]
    )

    try:
        metric.measure(empty_context_case)
    except ValueError as error:
        assert str(error) == "ContextualRelevancyMetric requires retrieval_context."
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "ContextualRelevancyMetric should raise ValueError "
            "when retrieval_context is empty."
        )

    print("\n========== CONTEXTUAL RELEVANCY TESTS PASSED ==========")


def print_contextual_recall_result(result):
    print("\n========== CONTEXTUAL RECALL EVALUATION ==========")
    print()
    print("Metric:", result.metric_name)
    print("Label:", result.label)
    print("Score:", result.score)
    print("Passed:", result.passed)

    print("\nDiagnostics:")
    print(result.diagnostics)

    print("\nProbabilities:")
    print(result.details["probabilities"])


def test_contextual_recall():
    print("\n========== CONTEXTUAL RECALL METRIC TESTS ==========")

    # Test 1 - Basic Contextual Recall
    print("\n--- Contextual Recall Test 1: Basic ---")

    metric = ContextualRecallMetric(model=shared_model)

    result = metric.measure(contextual_recall_case)

    assert result.metric_name == "contextual_recall"
    assert result.label in {"high_recall", "partial_recall", "low_recall"}
    assert "probabilities" in result.details

    print_contextual_recall_result(result)

    # Test 2 - Partial Recall
    print("\n--- Contextual Recall Test 2: Partial recall ---")

    metric = ContextualRecallMetric(model=shared_model)

    result = metric.measure(contextual_recall_partial_case)

    assert result.label in {"high_recall", "partial_recall", "low_recall"}

    print_contextual_recall_result(result)

    # Test 3 - ExpectedUtility
    print("\n--- Contextual Recall Test 3: ExpectedUtility ---")

    metric = ContextualRecallMetric(
        scoring_strategy=ExpectedUtility(
            utilities={
                "high_recall": 1.0,
                "partial_recall": 0.5,
                "low_recall": 0.0,
            }
        ),
        model=shared_model
    )

    result = metric.measure(contextual_recall_case)

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    print_contextual_recall_result(result)

    # Test 4 - ProbabilityOf
    print("\n--- Contextual Recall Test 4: ProbabilityOf('high_recall') ---")

    metric = ContextualRecallMetric(
        scoring_strategy=ProbabilityOf("high_recall"),
        model=shared_model
    )

    result = metric.measure(contextual_recall_case)

    assert result.score == result.details["probabilities"]["high_recall"]

    print_contextual_recall_result(result)

    # Test 5 - Margin diagnostic
    print("\n--- Contextual Recall Test 5: Margin diagnostic ---")

    metric = ContextualRecallMetric(
        scoring_strategy=MaxProbability(),
        diagnostics=[
            Margin()
        ],
        model=shared_model
    )

    result = metric.measure(contextual_recall_case)

    assert "Margin" in result.diagnostics
    assert result.diagnostics["Margin"] >= 0

    print_contextual_recall_result(result)

    # Test 6 - ScoreThreshold
    print("\n--- Contextual Recall Test 6: ScoreThreshold ---")

    metric = ContextualRecallMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=ScoreThreshold(
            threshold=0.5
        ),
        model=shared_model
    )

    result = metric.measure(contextual_recall_case)

    assert result.passed in (True, False)

    print_contextual_recall_result(result)

    # Test 7 - LabelMatch
    print("\n--- Contextual Recall Test 7: LabelMatch ---")

    metric = ContextualRecallMetric(
        passing_strategy=LabelMatch(
            expected_label="high_recall"
        ),
        model=shared_model
    )

    result = metric.measure(contextual_recall_case)

    assert result.passed in (True, False)

    print_contextual_recall_result(result)

    # Test 8 - Missing expected output
    print("\n--- Contextual Recall Test 8: Missing expected_output ---")

    metric = ContextualRecallMetric(model=shared_model)

    missing_expected_case = LLMEvaluationCase(
        input="What are the refund conditions?",
        actual_output="Refunds are available within 30 days.",
        expected_output=None,
        retrieval_context=[
            "Customers can request a refund within 30 days."
        ]
    )

    try:
        metric.measure(missing_expected_case)
    except ValueError as error:
        assert str(error) == "ContextualRecallMetric requires expected_output."
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "ContextualRecallMetric should raise ValueError "
            "when expected_output is missing."
        )

    # Test 9 - Empty retrieval context
    print("\n--- Contextual Recall Test 9: Empty retrieval_context ---")

    metric = ContextualRecallMetric(model=shared_model)

    empty_context_case = LLMEvaluationCase(
        input="What are the refund conditions?",
        actual_output="Refunds are available within 30 days.",
        expected_output="Customers can request a refund within 30 days.",
        retrieval_context=[]
    )

    try:
        metric.measure(empty_context_case)
    except ValueError as error:
        assert str(error) == "ContextualRecallMetric requires retrieval_context."
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "ContextualRecallMetric should raise ValueError "
            "when retrieval_context is empty."
        )

    print("\n========== CONTEXTUAL RECALL TESTS PASSED ==========")


def print_contextual_precision_result(result):
    print("\n========== CONTEXTUAL PRECISION EVALUATION ==========")
    print()
    print("Metric:", result.metric_name)
    print("Label:", result.label)
    print("Score:", result.score)
    print("Passed:", result.passed)

    print("\nDiagnostics:")
    print(result.diagnostics)

    print("\nProbabilities:")
    print(result.details["probabilities"])


def test_contextual_precision():
    print("\n========== CONTEXTUAL PRECISION METRIC TESTS ==========")

    # Test 1 - High Precision Retrieval
    print("\n--- Contextual Precision Test 1: High precision ---")

    metric = ContextualPrecisionMetric(model=shared_model)

    result = metric.measure(contextual_precision_case)

    assert result.metric_name == "contextual_precision"
    assert result.label in {"high_precision", "partial_precision", "low_precision"}
    assert "probabilities" in result.details

    print_contextual_precision_result(result)

    # Test 2 - Poor Precision / Relevant Information Buried
    print("\n--- Contextual Precision Test 2: Buried relevant information ---")

    metric = ContextualPrecisionMetric(model=shared_model)

    result = metric.measure(contextual_precision_buried_case)

    assert result.label in {"high_precision", "partial_precision", "low_precision"}

    print_contextual_precision_result(result)

    # Test 3 - ExpectedUtility
    print("\n--- Contextual Precision Test 3: ExpectedUtility ---")

    metric = ContextualPrecisionMetric(
        scoring_strategy=ExpectedUtility(
            utilities={
                "high_precision": 1.0,
                "partial_precision": 0.5,
                "low_precision": 0.0,
            }
        ),
        model=shared_model
    )

    result = metric.measure(contextual_precision_case)

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    print_contextual_precision_result(result)

    # Test 4 - ProbabilityOf
    print("\n--- Contextual Precision Test 4: ProbabilityOf('high_precision') ---")

    metric = ContextualPrecisionMetric(
        scoring_strategy=ProbabilityOf("high_precision"),
        model=shared_model
    )

    result = metric.measure(contextual_precision_case)

    assert result.score == result.details["probabilities"]["high_precision"]

    print_contextual_precision_result(result)

    # Test 5 - Margin diagnostic
    print("\n--- Contextual Precision Test 5: Margin diagnostic ---")

    metric = ContextualPrecisionMetric(
        scoring_strategy=MaxProbability(),
        diagnostics=[
            Margin()
        ],
        model=shared_model
    )

    result = metric.measure(contextual_precision_case)

    assert "Margin" in result.diagnostics
    assert result.diagnostics["Margin"] >= 0

    print_contextual_precision_result(result)

    # Test 6 - ScoreThreshold
    print("\n--- Contextual Precision Test 6: ScoreThreshold ---")

    metric = ContextualPrecisionMetric(
        scoring_strategy=MaxProbability(),
        passing_strategy=ScoreThreshold(
            threshold=0.5
        ),
        model=shared_model
    )

    result = metric.measure(contextual_precision_case)

    assert result.passed in (True, False)

    print_contextual_precision_result(result)

    # Test 7 - LabelMatch
    print("\n--- Contextual Precision Test 7: LabelMatch ---")

    metric = ContextualPrecisionMetric(
        passing_strategy=LabelMatch(
            expected_label="high_precision"
        ),
        model=shared_model
    )

    result = metric.measure(contextual_precision_case)

    assert result.passed in (True, False)

    print_contextual_precision_result(result)

    # Test 8 - Missing expected output
    print("\n--- Contextual Precision Test 8: Missing expected_output ---")

    metric = ContextualPrecisionMetric(model=shared_model)

    missing_expected_case = LLMEvaluationCase(
        input="What are the refund conditions?",
        actual_output="Refunds are available within 30 days.",
        expected_output=None,
        retrieval_context=[
            "Customers can request a refund within 30 days."
        ]
    )

    try:
        metric.measure(missing_expected_case)
    except ValueError as error:
        assert str(error) == "ContextualPrecisionMetric requires expected_output."
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "ContextualPrecisionMetric should raise ValueError "
            "when expected_output is missing."
        )

    # Test 9 - Empty retrieval context
    print("\n--- Contextual Precision Test 9: Empty retrieval_context ---")

    metric = ContextualPrecisionMetric(model=shared_model)

    empty_context_case = LLMEvaluationCase(
        input="What are the refund conditions?",
        actual_output="Refunds are available within 30 days.",
        expected_output="Customers can request a refund within 30 days.",
        retrieval_context=[]
    )

    try:
        metric.measure(empty_context_case)
    except ValueError as error:
        assert str(error) == "ContextualPrecisionMetric requires retrieval_context."
        print("\nRaised expected ValueError:", error)
    else:
        raise AssertionError(
            "ContextualPrecisionMetric should raise ValueError "
            "when retrieval_context is empty."
        )

    print("\n========== CONTEXTUAL PRECISION TESTS PASSED ==========")


def run_all():
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


    # ---------------------------------------------------------------
    # Contextual Relevancy tests
    # ---------------------------------------------------------------

    test_contextual_relevancy()


    # ---------------------------------------------------------------
    # Contextual Recall tests
    # ---------------------------------------------------------------

    test_contextual_recall()


    # ---------------------------------------------------------------
    # Contextual Precision tests
    # ---------------------------------------------------------------

    test_contextual_precision()


    print("\n========== SUMMARY ==========")
    print("Correctness          [PASS]")
    print("Relevance            [PASS]")
    print("Faithfulness         [PASS]")
    print("ContextualRelevancy  [PASS]")
    print("ContextualRecall     [PASS]")
    print("ContextualPrecision  [PASS]")

    print("\n========== ALL TESTS PASSED ==========")


# Running the file directly keeps the original behaviour (every metric).
# Passing a metric name runs only that metric's tests, so the RAG metrics
# can be validated in isolation without executing the other suites.
if __name__ == "__main__":
    import sys

    selection = sys.argv[1] if len(sys.argv) > 1 else "all"

    if selection == "contextual_relevancy":
        test_contextual_relevancy()
    elif selection == "contextual_recall":
        test_contextual_recall()
    elif selection == "contextual_precision":
        test_contextual_precision()
    else:
        run_all()
