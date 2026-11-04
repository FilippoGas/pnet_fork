"""Tests for hierarchical stratified class balancing."""

import numpy as np
import pandas as pd

from util import util


def make_synthetic_data():
    """Create a small labeled dataset with known ethnicity and cancer-type strata."""
    samples = [f"s{i}" for i in range(100)]
    labels = pd.DataFrame({"target": [1] * 20 + [0] * 80}, index=samples)

    ethnicity = ["A"] * 40 + ["B"] * 35 + ["C"] * 25
    tumor_type = (
        ["X"] * 20
        + ["X"] * 10 + ["Y"] * 10
        + ["X"] * 10 + ["Y"] * 25
        + ["X"] * 5 + ["Y"] * 20
    )
    strata = pd.DataFrame({"ethnicity": ethnicity, "tumor_type": tumor_type}, index=samples)
    return samples, labels, strata


def test_balanced_counts_and_membership():
    """Check equal class sizes, unique samples, and valid pool membership."""
    samples, labels, strata = make_synthetic_data()
    balanced = util.balance_split(samples, labels, strata=strata, random_state=0)

    balanced_labels = labels.loc[balanced, "target"]
    assert len(balanced) == 40
    assert len(set(balanced)) == 40
    assert balanced_labels.sum() == 20
    assert (balanced_labels == 0).sum() == 20
    assert set(samples[:20]).issubset(balanced)
    assert set(balanced).issubset(samples)


def test_ethnicity_and_tumor_proportions():
    """Check that the sampled majority preserves the expected strata proportions."""
    samples, labels, strata = make_synthetic_data()
    balanced = util.balance_split(samples, labels, strata=strata, random_state=0)

    negatives = [s for s in balanced if labels.loc[s, "target"] == 0]
    ethnicity_counts = strata.loc[negatives, "ethnicity"].value_counts()
    tumor_counts = strata.loc[negatives, "tumor_type"].value_counts()

    assert ethnicity_counts.to_dict() == {"B": 9, "C": 6, "A": 5}
    assert abs(tumor_counts["X"] - 6) <= 1
    assert abs(tumor_counts["Y"] - 14) <= 1


def test_deterministic_with_random_state():
    """Check that the seed controls member selection while counts stay fixed."""
    samples, labels, strata = make_synthetic_data()
    first = util.balance_split(samples, labels, strata=strata, random_state=7)
    second = util.balance_split(samples, labels, strata=strata, random_state=7)
    third = util.balance_split(samples, labels, strata=strata, random_state=8)

    assert first == second
    assert first != third
    assert len(first) == len(third)


def test_backward_compatible_without_strata():
    """Check that omitting strata still produces balanced uniform sampling."""
    samples, labels, _ = make_synthetic_data()
    balanced = util.balance_split(samples, labels, random_state=0)
    balanced_labels = labels.loc[balanced, "target"]

    assert len(balanced) == 40
    assert balanced_labels.sum() == 20
    assert (balanced_labels == 0).sum() == 20


def test_series_strata_and_missing_values():
    """Check single-dimension Series strata and missing-value handling."""
    samples, labels, strata = make_synthetic_data()
    ethnicity = strata["ethnicity"].copy()
    ethnicity.iloc[20:25] = np.nan

    balanced = util.balance_split(samples, labels, strata=ethnicity, random_state=0)
    balanced_labels = labels.loc[balanced, "target"]

    assert len(balanced) == 40
    assert balanced_labels.sum() == 20
    assert (balanced_labels == 0).sum() == 20


def test_empty_minority_and_small_majority():
    """Check empty-minority and majority-smaller-than-minority edge cases."""
    samples = [f"s{i}" for i in range(8)]
    labels = pd.DataFrame({"target": [0] * 8}, index=samples)
    strata = pd.DataFrame({"ethnicity": ["A"] * 4 + ["B"] * 4,
                           "tumor_type": ["X"] * 4 + ["Y"] * 4}, index=samples)

    assert util.balance_split(samples, labels, strata=strata, random_state=0) == []

    labels = pd.DataFrame({"target": [1] * 5 + [0] * 3}, index=samples)
    balanced = util.balance_split(samples, labels, strata=strata, random_state=0)
    assert len(balanced) == 8
    assert set(balanced) == set(samples)


def main():
    test_balanced_counts_and_membership()
    test_ethnicity_and_tumor_proportions()
    test_deterministic_with_random_state()
    test_backward_compatible_without_strata()
    test_series_strata_and_missing_values()
    test_empty_minority_and_small_majority()
    print("All balance_split tests passed!")


if __name__ == "__main__":
    main()
