from __future__ import annotations

import numpy as np
import pytest
import torch

from dogsai.narrate import (
    BOS,
    EOS,
    PAD,
    VOICE_KINDS,
    CausalSelfAttention,
    DogNarrator,
    NarrationDataset,
    NarrationExample,
    NarratorConfig,
    NarratorTrainConfig,
    WordTokenizer,
    condition_dim,
    condition_vector,
    load_narrator,
    narrate,
    save_narrator,
    train_narrator,
)

CAPTIONS = ["Dog is eating.", "Dog is yawning.", "Dog is playing.", "Dog bites rope."]


class TestTokenizer:
    def test_round_trips_a_caption(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        assert tokenizer.decode(tokenizer.encode("Dog is eating.")) == "Dog is eating."

    def test_specials_occupy_the_first_slots(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        assert tokenizer.itos[PAD] == "<pad>"
        assert tokenizer.encode("dog")[0] == BOS
        assert tokenizer.encode("dog")[-1] == EOS

    def test_punctuation_is_its_own_token(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        assert "." in tokenizer.stoi
        # "eating" and "eating." must not be separate vocabulary entries
        assert "eating." not in tokenizer.stoi

    def test_unknown_words_map_to_unk_without_raising(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        decoded = tokenizer.decode(tokenizer.encode("Dog is somersaulting."))
        assert "<unk>" in decoded.lower() or "unk" in decoded.lower()

    def test_decode_does_not_space_before_punctuation(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        assert " ." not in tokenizer.decode(tokenizer.encode("Dog is eating."))

    def test_min_count_prunes_rare_words(self):
        big = WordTokenizer.build(CAPTIONS, min_count=1)
        small = WordTokenizer.build(CAPTIONS, min_count=2)
        assert len(small) < len(big)

    def test_save_and_load(self, tmp_path):
        tokenizer = WordTokenizer.build(CAPTIONS)
        tokenizer.save(tmp_path / "v.json")
        assert WordTokenizer.load(tmp_path / "v.json").itos == tokenizer.itos


class TestConditionVector:
    def test_length_matches_the_declared_dimension(self):
        vector = condition_vector(np.zeros(5), None, 7.0)
        assert len(vector) == condition_dim(5)

    def test_encodes_behaviour_scores_first(self):
        scores = np.array([0.1, 0.9, 0.0])
        vector = condition_vector(scores, None, 0.0)
        assert np.allclose(vector[:3], scores)

    def test_voice_counts_are_compressed(self):
        """Thirty barks must not be 10x the feature value of three."""
        few = condition_vector(np.zeros(2), {"kinds": {"bark": 3}}, 5.0)
        many = condition_vector(np.zeros(2), {"kinds": {"bark": 30}}, 5.0)
        index = 2 + VOICE_KINDS.index("bark")
        assert many[index] > few[index]
        assert many[index] < few[index] * 3

    def test_audio_summary_fields_land_in_the_tail(self):
        vector = condition_vector(
            np.zeros(2), {"arousal": 0.8, "valence": -0.5, "vocal_fraction": 0.25}, 10.0
        )
        assert vector[-4] == pytest.approx(0.8)
        assert vector[-3] == pytest.approx(-0.5)
        assert vector[-2] == pytest.approx(0.25)

    def test_missing_audio_is_zeroed_not_an_error(self):
        vector = condition_vector(np.ones(3), None, 0.0)
        assert np.allclose(vector[3:], 0.0)

    def test_is_finite_for_extreme_duration(self):
        assert np.all(np.isfinite(condition_vector(np.zeros(3), None, 1e9)))


class TestAttention:
    def test_output_shape(self):
        layer = CausalSelfAttention(32, 4).eval()
        assert layer(torch.randn(2, 7, 32)).shape == (2, 7, 32)

    def test_is_causal(self):
        """A later token must not influence an earlier position's output."""
        layer = CausalSelfAttention(32, 4).eval()
        x = torch.randn(1, 6, 32)
        with torch.no_grad():
            base = layer(x)
            changed = x.clone()
            changed[:, -1] += 10.0        # perturb only the final token
            after = layer(changed)
        assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-5)
        assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-3)

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(ValueError):
            CausalSelfAttention(30, 4)


class TestNarratorModel:
    def config(self, **kwargs) -> NarratorConfig:
        defaults = dict(vocab_size=20, condition_dim=17, dim=32, depth=2, heads=4, max_len=16)
        defaults.update(kwargs)
        return NarratorConfig(**defaults)

    def test_forward_shape_excludes_the_prefix(self):
        model = DogNarrator(self.config()).eval()
        logits = model(torch.randn(3, 17), torch.zeros(3, 9, dtype=torch.long))
        assert logits.shape == (3, 9, 20)

    def test_embeddings_are_tied_to_the_output_head(self):
        model = DogNarrator(self.config())
        assert model.head.weight is model.tokens.weight

    def test_conditioning_changes_the_output(self):
        """If it ignored the condition it would not be a conditioned model."""
        model = DogNarrator(self.config()).eval()
        tokens = torch.tensor([[BOS, 5, 6]])
        with torch.no_grad():
            a = model(torch.zeros(1, 17), tokens)
            b = model(torch.ones(1, 17) * 3, tokens)
        assert not torch.allclose(a, b, atol=1e-4)

    def test_generate_returns_token_ids_and_stops(self):
        model = DogNarrator(self.config()).eval()
        ids = model.generate(torch.randn(17), max_new_tokens=8)
        assert isinstance(ids, list)
        assert len(ids) <= 8
        assert EOS not in ids
        assert PAD not in ids

    def test_greedy_generation_is_deterministic(self):
        model = DogNarrator(self.config()).eval()
        condition = torch.randn(17)
        assert model.generate(condition) == model.generate(condition)

    def test_sampling_can_differ(self):
        torch.manual_seed(0)
        model = DogNarrator(self.config()).eval()
        condition = torch.randn(17)
        draws = {
            tuple(model.generate(condition, temperature=1.5, top_k=5)) for _ in range(8)
        }
        assert len(draws) > 1

    def test_gradients_reach_every_parameter(self):
        model = DogNarrator(self.config())
        logits = model(torch.randn(2, 17), torch.ones(2, 5, dtype=torch.long))
        logits.sum().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []

    def test_stays_small(self):
        model = DogNarrator(NarratorConfig(vocab_size=64, condition_dim=17))
        assert model.num_parameters() < 2e6


class TestNarrationDataset:
    def test_pads_and_shifts_for_teacher_forcing(self):
        tokenizer = WordTokenizer.build(CAPTIONS)
        dataset = NarrationDataset(
            [NarrationExample(condition_vector(np.zeros(5), None, 3.0), CAPTIONS[0])],
            tokenizer, max_len=12,
        )
        item = dataset[0]
        assert item["tokens"].shape == item["targets"].shape == (12,)
        # targets are the inputs shifted by one
        assert int(item["tokens"][0]) == BOS
        assert int(item["targets"][0]) == int(item["tokens"][1])

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            NarrationDataset([], WordTokenizer.build(CAPTIONS))


class TestTrainingAndRoundTrip:
    def test_learns_a_deterministic_caption_mapping(self, tmp_path):
        """Five behaviours, five captions: the model must reach it exactly.

        This is a low bar by design — it is the same bar the real dataset sets, and
        the point of the test is that the plumbing (conditioning, teacher forcing,
        generation, save/load) is correct end to end.
        """
        behaviours = ["chewing", "eating_drinking", "eliminating", "playing", "yawning"]
        captions = [f"Dog is {b.replace('_', ' ')}." for b in behaviours]
        tokenizer = WordTokenizer.build(captions)

        examples = []
        for _ in range(12):
            for i, caption in enumerate(captions):
                scores = np.full(len(behaviours), 0.03, dtype=np.float32)
                scores[i] = 0.88
                examples.append(
                    NarrationExample(condition_vector(scores, None, 7.0), caption)
                )

        model, result = train_narrator(
            examples, examples[: len(captions)], tokenizer,
            NarratorConfig(vocab_size=len(tokenizer),
                           condition_dim=condition_dim(len(behaviours)),
                           dim=64, depth=2, heads=4, max_len=16),
            NarratorTrainConfig(epochs=60, batch_size=16, lr=3e-3, warmup_epochs=2),
            verbose=False,
        )
        assert result.val_perplexity < 1.5
        assert result.exact_match >= 0.8

        # every behaviour maps to its own caption
        for i, behaviour in enumerate(behaviours):
            scores = np.full(len(behaviours), 0.03, dtype=np.float32)
            scores[i] = 0.88
            text = narrate(model, tokenizer, scores, None, 7.0)
            assert behaviour.replace("_", " ") in text.lower(), (behaviour, text)

        path = save_narrator(tmp_path / "n.pt", model, tokenizer, behaviours, result)
        restored, restored_tokenizer, restored_behaviours, metrics = load_narrator(path)
        assert restored_behaviours == behaviours
        assert restored_tokenizer.itos == tokenizer.itos
        assert metrics["exact_match"] == result.exact_match

        condition = torch.from_numpy(
            condition_vector(np.eye(5, dtype=np.float32)[0], None, 7.0)
        )
        assert restored.generate(condition) == model.generate(condition)

    def test_training_history_is_recorded(self):
        behaviours = ["a", "b"]
        tokenizer = WordTokenizer.build(["Dog is a.", "Dog is b."])
        examples = [
            NarrationExample(condition_vector(np.eye(2, dtype=np.float32)[i], None, 3.0),
                             f"Dog is {b}.")
            for i, b in enumerate(behaviours)
        ] * 8
        _, result = train_narrator(
            examples, examples[:2], tokenizer,
            NarratorConfig(vocab_size=len(tokenizer), condition_dim=condition_dim(2),
                           dim=32, depth=1, heads=2, max_len=12),
            NarratorTrainConfig(epochs=5, batch_size=8, warmup_epochs=1),
            verbose=False,
        )
        assert len(result.history) == 5
        assert all("train_loss" in r for r in result.history)
        assert "perplexity" in result.summary()
