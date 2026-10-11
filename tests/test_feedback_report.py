"""Feedback AUC counts score rankings and gives ties half credit."""

import pytest

from scripts.feedback_report import auc


@pytest.mark.parametrize(
    ("positives", "negatives", "expected"),
    [
        pytest.param([0.9, 0.8], [0.2, 0.1, 0.3], 1.0, id="perfect-separation"),
        pytest.param([0.2, 0.1, 0.3], [0.9, 0.8], 0.0, id="reversed"),
        pytest.param([0.5, 0.5], [0.5, 0.5, 0.5], 0.5, id="all-ties"),
    ],
)
def test_auc_rankings(positives, negatives, expected):
    assert auc(positives, negatives) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("positives", "negatives"),
    [
        pytest.param([], [0.5], id="no-positives"),
        pytest.param([0.5], [], id="no-negatives"),
        pytest.param([], [], id="both-empty"),
    ],
)
def test_auc_requires_both_classes(positives, negatives):
    assert auc(positives, negatives) is None


def test_auc_mixed_rankings_with_a_tie():
    # 0.8 wins all three comparisons; 0.4 wins one, ties one and loses one.
    # Four wins plus half a tie, out of six pairs, gives 0.75.
    assert auc([0.8, 0.4], [0.4, 0.6, 0.2]) == pytest.approx(0.75)
