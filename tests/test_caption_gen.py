from __future__ import annotations

import pytest

from dogsai.caption_gen import (
    GeneratedCaption,
    compose_caption,
    corpus_stats,
    generate_captions,
    load_corpus,
    save_corpus,
)
from dogsai.dataset import discover_split


class TestComposeCaption:
    def test_deterministic_by_seed(self):
        a = compose_caption(["playing"], 5.0, None, seed=42)
        b = compose_caption(["playing"], 5.0, None, seed=42)
        assert a == b

    def test_different_seeds_usually_differ(self):
        variants = {compose_caption(["playing"], 5.0, None, seed=s) for s in range(20)}
        assert len(variants) > 1

    def test_mentions_the_behaviour(self):
        # Several synonyms exist per behaviour; check across seeds that the
        # composed sentence always uses one of the declared phrases for it.
        from dogsai.caption_gen import _ACTION_PHRASES

        phrases = _ACTION_PHRASES["eating_drinking"]
        for seed in range(15):
            text = compose_caption(["eating_drinking"], 5.0, None, seed=seed).lower()
            assert any(p in text for p in phrases), text

    def test_no_behaviour_still_produces_a_sentence(self):
        text = compose_caption([], 3.0, None, seed=1)
        assert text and text.endswith(".")

    def test_ends_with_a_period(self):
        for seed in range(10):
            assert compose_caption(["playing"], 4.0, None, seed=seed).endswith(".")

    def test_starts_with_a_capital(self):
        text = compose_caption(["playing"], 4.0, None, seed=3)
        assert text[0].isupper()

    def test_zero_duration_does_not_crash(self):
        assert compose_caption(["sitting"], 0.0, None, seed=1)


class TestGenerateCaptions:
    def test_produces_one_caption_per_annotation(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        captions = generate_captions(annotations, with_audio=False, verbose=False)
        assert len(captions) == len(annotations)
        assert all(isinstance(c, GeneratedCaption) for c in captions)

    def test_is_reproducible_across_runs(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        a = generate_captions(annotations, with_audio=False, verbose=False)
        b = generate_captions(annotations, with_audio=False, verbose=False)
        assert [c.caption for c in a] == [c.caption for c in b]

    def test_captions_are_meaningfully_diverse(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        captions = generate_captions(annotations, with_audio=False, verbose=False)
        stats = corpus_stats(captions)
        # The whole point versus the source dataset's 5 fixed strings.
        assert stats["distinct"] > 5
        assert stats["vocabulary"] > 20

    def test_to_annotation_round_trips_the_caption(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        item = generate_captions(annotations[:1], with_audio=False, verbose=False)[0]
        restored = item.to_annotation(0.0, 2.0)
        assert restored.meta["caption"] == item.caption
        assert restored.meta["generated"] is True


class TestCorpusStats:
    def test_all_identical_captions_have_zero_diversity(self):
        items = [
            GeneratedCaption("v.mp4", "Same.", ["x"], 1.0, {}, "g") for _ in range(5)
        ]
        stats = corpus_stats(items)
        assert stats["distinct"] == 1
        assert stats["distinct_ratio"] == pytest.approx(0.2)

    def test_empty_corpus_does_not_crash(self):
        stats = corpus_stats([])
        assert stats["captions"] == 0


class TestSaveLoadCorpus:
    def test_round_trips(self, tmp_path, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:5]
        original = generate_captions(annotations, with_audio=False, verbose=False)
        path = save_corpus(tmp_path / "c.jsonl", original)
        restored = load_corpus(path)
        assert [c.caption for c in restored] == [c.caption for c in original]
        assert [c.labels for c in restored] == [c.labels for c in original]

    def test_empty_lines_are_skipped(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text('{"video":"a.mp4","caption":"Hi.","labels":["x"],"duration":1.0,"group":"g","audio":{}}\n\n')
        assert len(load_corpus(path)) == 1
