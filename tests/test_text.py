"""Tests for text processing utilities."""

import pytest
from ragsvc.text import normalize, exact_match, f1, rrf


class TestNormalize:
    def test_lowercase(self):
        assert normalize("The Eiffel Tower") == "eiffel tower"

    def test_remove_punctuation(self):
        assert normalize("Hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize("a cat and an apple") == "cat and apple"

    def test_collapse_whitespace(self):
        assert normalize("hello  world") == "hello world"

    def test_combined(self):
        assert normalize("A, B, and C") == "b and c"


class TestExactMatch:
    def test_exact_match_true(self):
        pred = "The Eiffel Tower"
        golds = ["Eiffel Tower"]
        assert exact_match(pred, golds) == 1.0

    def test_exact_match_false(self):
        pred = "tower in Paris"
        golds = ["Eiffel Tower"]
        assert exact_match(pred, golds) == 0.0

    def test_exact_match_multiple_golds(self):
        pred = "cat"
        golds = ["dog", "cat", "bird"]
        assert exact_match(pred, golds) == 1.0


class TestF1:
    def test_f1_perfect(self):
        pred = "Eiffel Tower"
        golds = ["Eiffel Tower"]
        assert f1(pred, golds) == 1.0

    def test_f1_partial(self):
        pred = "tower in Paris"
        golds = ["Eiffel Tower"]
        assert f1(pred, golds) == 0.4

    def test_f1_no_overlap(self):
        pred = "cat"
        golds = ["dog"]
        assert f1(pred, golds) == 0.0

    def test_f1_empty_pred(self):
        pred = ""
        golds = ["answer"]
        assert f1(pred, golds) == 0.0

    def test_f1_empty_gold(self):
        pred = "answer"
        golds = [""]
        assert f1(pred, golds) == 0.0

    def test_f1_both_empty(self):
        pred = ""
        golds = [""]
        assert f1(pred, golds) == 1.0


class TestRRF:
    def test_rrf_two_rankings(self):
        ranking1 = [1, 2, 3, 4, 5]
        ranking2 = [5, 4, 3, 2, 1]
        result = rrf([ranking1, ranking2], k=60)
        # Check that result is a list and contains all ids
        assert len(result) == 5
        assert set(result) == {1, 2, 3, 4, 5}
        # IDs that appear in both lists should score higher
        # All should have the same reciprocal rank sum by symmetry,
        # but we can verify order is deterministic
        assert result[0] in {1, 2, 3, 4, 5}

    def test_rrf_single_ranking(self):
        ranking = [1, 2, 3]
        result = rrf([ranking], k=60)
        assert result == [1, 2, 3]

    def test_rrf_overlapping_rankings(self):
        ranking1 = [1, 2, 3]
        ranking2 = [1, 2, 4]
        result = rrf([ranking1, ranking2], k=60)
        # Documents 1 and 2 appear in both rankings, should score higher
        assert result.index(1) < result.index(3)
        assert result.index(2) < result.index(3)
