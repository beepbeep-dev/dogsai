from __future__ import annotations

import pytest
import torch

from dogsai.chat import (
    INTENTS,
    ChatConfig,
    ChatDataset,
    DogChat,
    DogState,
    ask,
    build_tokenizer,
    compose_answer,
    dialogue_stats,
    generate_dialogues,
    load_chat,
    save_chat,
    state_dim,
    train_chat,
)
from dogsai.narrate import BOS, EOS, PAD

BEHAVIOURS = ["chewing", "eating_drinking", "eliminating", "playing", "yawning"]


class TestDogState:
    def test_growl_outranks_the_posture_it_happened_during(self):
        state = DogState("playing", "growl", 0.7, -0.3)
        assert state.keys()[0] == "growl"

    def test_silent_state_falls_back_to_behaviour(self):
        state = DogState("playing", "", 0.5, 0.0)
        assert state.keys()[0] == "playing"

    def test_valence_bands_are_mutually_exclusive_labels(self):
        assert "positive_low" in DogState("x", "", 0.1, 0.5).keys()
        assert "negative_low" in DogState("x", "", 0.1, -0.5).keys()
        assert "neutral" in DogState("x", "", 0.1, 0.0).keys()

    def test_default_is_always_a_fallback_key(self):
        assert DogState("anything", "", 0.5, 0.0).keys()[-1] == "default"

    def test_vector_has_the_declared_dimension(self):
        state = DogState("playing", "growl", 0.5, -0.2)
        assert len(state.vector(BEHAVIOURS)) == state_dim(len(BEHAVIOURS))

    def test_vector_encodes_the_correct_voice_slot(self):
        from dogsai.narrate import VOICE_KINDS

        state = DogState("playing", "growl", 0.5, -0.2)
        vector = state.vector(BEHAVIOURS)
        voice_start = len(BEHAVIOURS)
        assert vector[voice_start + VOICE_KINDS.index("growl")] == 1.0


class TestComposeAnswer:
    def test_deterministic_by_seed(self):
        state = DogState("playing", "growl", 0.7, -0.5)
        assert compose_answer(state, "worry", 7) == compose_answer(state, "worry", 7)

    def test_growl_worry_answer_advises_against_punishment(self):
        state = DogState("playing", "growl", 0.7, -0.5)
        text = compose_answer(state, "worry", 1)
        assert "not punish" in text.lower() or "do not punish" in text.lower()

    def test_yelp_hurt_answer_mentions_pain(self):
        state = DogState("standing", "yelp", 0.8, -0.7)
        text = compose_answer(state, "hurt", 1).lower()
        assert "pain" in text or "hurt" in text

    def test_unknown_intent_falls_back_gracefully(self):
        state = DogState("playing", "", 0.5, 0.2)
        text = compose_answer(state, "not_a_real_intent", 1)
        assert text  # no crash, some default text


class TestGenerateDialogues:
    def test_covers_every_declared_intent(self):
        examples = generate_dialogues(BEHAVIOURS, repeats=1, verbose=False)
        assert {e.intent for e in examples} == set(INTENTS)

    def test_is_reproducible(self):
        a = generate_dialogues(BEHAVIOURS[:2], repeats=1, verbose=False)
        b = generate_dialogues(BEHAVIOURS[:2], repeats=1, verbose=False)
        assert [e.answer for e in a] == [e.answer for e in b]

    def test_repeats_increase_phrasing_diversity(self):
        one = generate_dialogues(BEHAVIOURS[:1], repeats=1, verbose=False)
        two = generate_dialogues(BEHAVIOURS[:1], repeats=3, verbose=False)
        stats_one = dialogue_stats(one)
        stats_two = dialogue_stats(two)
        assert stats_two["distinct_answers"] >= stats_one["distinct_answers"]

    def test_dialogue_stats_reports_expected_fields(self):
        examples = generate_dialogues(BEHAVIOURS[:2], repeats=1, verbose=False)
        stats = dialogue_stats(examples)
        for key in ("pairs", "distinct_answers", "distinct_questions", "vocabulary", "intents"):
            assert key in stats
        assert stats["pairs"] == len(examples)


class TestChatModel:
    def config(self, **kwargs) -> ChatConfig:
        defaults = dict(vocab_size=40, state_dim=state_dim(2), dim=32, depth=2, heads=4, max_len=24)
        defaults.update(kwargs)
        return ChatConfig(**defaults)

    def test_forward_shape(self):
        model = DogChat(self.config()).eval()
        logits = model(torch.randn(3, state_dim(2)), torch.zeros(3, 10, dtype=torch.long))
        assert logits.shape == (3, 10, 40)

    def test_state_changes_the_output(self):
        model = DogChat(self.config()).eval()
        tokens = torch.tensor([[BOS, 5, 6, 4, 7]])
        with torch.no_grad():
            a = model(torch.zeros(1, state_dim(2)), tokens)
            b = model(torch.ones(1, state_dim(2)) * 2, tokens)
        assert not torch.allclose(a, b, atol=1e-4)

    def test_reply_stops_and_avoids_special_tokens(self):
        model = DogChat(self.config()).eval()
        ids = model.reply(torch.randn(state_dim(2)), [5, 6, 7], max_new_tokens=10)
        assert EOS not in ids and PAD not in ids

    def test_reply_is_deterministic_when_greedy(self):
        model = DogChat(self.config()).eval()
        state = torch.randn(state_dim(2))
        assert model.reply(state, [5, 6]) == model.reply(state, [5, 6])

    def test_embeddings_tied(self):
        model = DogChat(self.config())
        assert model.head.weight is model.tokens.weight

    def test_stays_reasonably_small(self):
        model = DogChat(ChatConfig(vocab_size=400, state_dim=15, dim=192, depth=4, heads=6))
        assert model.num_parameters() < 5e6


class TestChatDataset:
    def test_loss_mask_excludes_the_question(self):
        examples = generate_dialogues(BEHAVIOURS[:1], repeats=1, verbose=False)[:3]
        tokenizer = build_tokenizer(examples)
        dataset = ChatDataset(examples, tokenizer, BEHAVIOURS, max_len=40)
        item = dataset[0]
        # Everything before <sep> in targets must be PAD (masked out of the loss).
        from dogsai.chat import SEP

        sep_positions = (item["tokens"] == SEP).nonzero()
        if len(sep_positions):
            sep_at = int(sep_positions[0])
            assert (item["targets"][:sep_at] == PAD).all()

    def test_empty_examples_rejected(self):
        with pytest.raises(ValueError):
            ChatDataset([], build_tokenizer(generate_dialogues(BEHAVIOURS[:1], verbose=False)), BEHAVIOURS)


class TestTrainingRoundTrip:
    def test_learns_and_can_be_asked_a_question(self, tmp_path):
        examples = generate_dialogues(BEHAVIOURS, repeats=1, verbose=False)
        # Small, fast subset for the test.
        subset = [e for e in examples if e.intent in ("feeling", "hurt", "worry")][:300]
        tokenizer = build_tokenizer(subset)
        model, result = train_chat(
            subset, subset[:40], tokenizer, BEHAVIOURS,
            ChatConfig(vocab_size=len(tokenizer), state_dim=state_dim(len(BEHAVIOURS)),
                      dim=64, depth=2, heads=4, max_len=48),
            epochs=20, batch_size=32, lr=2e-3, verbose=False,
        )
        # A small subset for test speed, not a proxy for the real training run
        # (which reaches ppl ~1.06 on the full 21k-pair corpus over 14 epochs);
        # this just checks perplexity drops well below a random baseline.
        assert result.val_perplexity < len(tokenizer) / 4

        state = DogState("playing", "growl", 0.7, -0.5)
        answer = ask(model, tokenizer, BEHAVIOURS, state, "should i be worried")
        assert answer.text
        assert 0.0 <= answer.out_of_domain <= 1.0

        path = save_chat(tmp_path / "c.pt", model, tokenizer, BEHAVIOURS, result)
        restored, restored_tok, restored_behaviours, metrics = load_chat(path)
        assert restored_behaviours == BEHAVIOURS
        assert metrics["val_perplexity"] == result.val_perplexity

        q_ids = tokenizer.encode("should i be worried", add_special=False)
        assert restored.reply(torch.from_numpy(state.vector(BEHAVIOURS)), q_ids) == model.reply(
            torch.from_numpy(state.vector(BEHAVIOURS)), q_ids
        )


class TestAsk:
    def test_out_of_domain_score_reflects_unknown_words(self):
        examples = generate_dialogues(BEHAVIOURS[:1], repeats=1, verbose=False)[:20]
        tokenizer = build_tokenizer(examples)
        model = DogChat(ChatConfig(vocab_size=len(tokenizer), state_dim=state_dim(len(BEHAVIOURS)),
                                   dim=32, depth=1, heads=2, max_len=24)).eval()
        state = DogState("chewing", "", 0.3, 0.2)
        known = ask(model, tokenizer, BEHAVIOURS, state, "how are you feeling")
        gibberish = ask(model, tokenizer, BEHAVIOURS, state, "xqzwv florb splonk")
        assert gibberish.out_of_domain >= known.out_of_domain

    def test_uncertain_flag_follows_the_threshold(self):
        from dogsai.chat import ChatAnswer

        assert ChatAnswer("x", 0.5, "q").uncertain
        assert not ChatAnswer("x", 0.1, "q").uncertain
