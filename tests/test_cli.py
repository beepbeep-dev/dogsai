from __future__ import annotations

import pytest

from dogsai.cli import _load_config, build_parser, main


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


class TestConfigOverrides:
    def test_preset_alone_sets_resolution_and_width(self):
        config = _load_config(parse(["train", "--set", "model.preset=base"]))
        assert config.data.image_size == 192
        assert config.data.clip_frames == 24
        assert config.model.width == 1.0

    def test_explicit_override_beats_the_preset(self):
        """Regression: the preset used to be re-applied last and clobber these."""
        config = _load_config(parse([
            "train",
            "--set", "model.preset=nano",
            "--set", "data.image_size=112",
            "--set", "data.clip_frames=10",
        ]))
        assert config.model.preset == "nano"
        assert config.model.width == 0.5      # from the preset
        assert config.data.image_size == 112  # from the override
        assert config.data.clip_frames == 10

    def test_override_order_does_not_matter(self):
        a = _load_config(parse(["train", "--set", "data.image_size=96",
                                "--set", "model.preset=nano"]))
        b = _load_config(parse(["train", "--set", "model.preset=nano",
                                "--set", "data.image_size=96"]))
        assert a.data.image_size == b.data.image_size == 96

    def test_types_are_coerced_from_strings(self):
        config = _load_config(parse([
            "train",
            "--set", "train.epochs=7",
            "--set", "train.lr=0.0003",
            "--set", "train.amp=false",
            "--set", "data.train_scale=0.5,1.0",
        ]))
        assert config.train.epochs == 7 and isinstance(config.train.epochs, int)
        assert config.train.lr == pytest.approx(3e-4)
        assert config.train.amp is False
        assert config.data.train_scale == (0.5, 1.0)

    def test_bool_accepts_common_spellings(self):
        for text, expected in (("true", True), ("1", True), ("yes", True),
                               ("false", False), ("0", False), ("no", False)):
            config = _load_config(parse(["train", "--set", f"train.amp={text}"]))
            assert config.train.amp is expected, text

    def test_unknown_key_is_rejected(self):
        with pytest.raises(SystemExit):
            _load_config(parse(["train", "--set", "train.not_a_key=1"]))

    def test_unknown_section_is_rejected(self):
        with pytest.raises(SystemExit):
            _load_config(parse(["train", "--set", "nope.thing=1"]))

    def test_malformed_override_is_rejected(self):
        with pytest.raises(SystemExit):
            _load_config(parse(["train", "--set", "train.epochs"]))

    def test_data_root_and_out_dir_flags_apply(self):
        config = _load_config(parse([
            "train", "--data-root", "/x/data", "--out-dir", "/x/run", "--task", "multiclass",
        ]))
        assert config.data.root == "/x/data"
        assert config.train.out_dir == "/x/run"
        assert config.task == "multiclass"

    def test_config_file_is_honoured(self, tmp_path):
        from dogsai.config import Config

        source = Config().apply_preset()
        source.train.epochs = 99
        source.save(tmp_path / "c.json")
        config = _load_config(parse(["train", "--config", str(tmp_path / "c.json")]))
        assert config.train.epochs == 99

    def test_set_overrides_a_config_file(self, tmp_path):
        from dogsai.config import Config

        source = Config().apply_preset()
        source.train.epochs = 99
        source.save(tmp_path / "c.json")
        config = _load_config(parse([
            "train", "--config", str(tmp_path / "c.json"), "--set", "train.epochs=3",
        ]))
        assert config.train.epochs == 3


class TestParser:
    def test_every_subcommand_is_wired_to_a_function(self):
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions if hasattr(action, "choices") and action.choices
        )
        assert set(subparsers.choices) >= {
            "synth", "prepare", "audit", "train", "eval", "predict",
            "info", "export", "behaviours", "fetch", "feeling",
        }
        for name, sub in subparsers.choices.items():
            assert sub.get_default("func") is not None, name

    def test_a_command_is_required(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])


class TestCommands:
    def test_behaviours_lists_the_taxonomy(self, capsys):
        assert main(["behaviours"]) == 0
        out = capsys.readouterr().out
        assert "posture" in out and "actions" in out
        assert "tail_wagging" in out and "yawning" in out

    def test_fetch_with_no_argument_lists_datasets(self, capsys):
        assert main(["fetch"]) == 0
        out = capsys.readouterr().out
        assert "dog-behaviour" in out
        assert "licence" in out

    def test_fetch_rejects_an_unknown_dataset(self, capsys):
        assert main(["fetch", "not_a_dataset"]) == 1

    def test_audit_on_a_missing_root_fails_cleanly(self, capsys):
        assert main(["audit", "--data-root", "/nonexistent/path"]) == 1
        assert "no splits found" in capsys.readouterr().err

    def test_audit_reports_a_clean_dataset(self, mini_dataset, capsys):
        code = main([
            "audit", "--data-root", str(mini_dataset),
            "--no-duplicate-check", "--any-label",
        ])
        out = capsys.readouterr().out
        assert code == 0
        assert "class balance" in out
        assert "no leakage detected" in out

    def test_audit_strict_fails_on_errors(self, tmp_path, mini_dataset):
        from dogsai.dataset import Annotation, discover_split, save_annotations

        annotations = discover_split(mini_dataset, "train")
        annotations.append(Annotation("/nope/missing.mp4", ["running"]))
        save_annotations(tmp_path / "train.jsonl", annotations)
        code = main([
            "audit", "--data-root", str(tmp_path), "--splits", "train",
            "--no-duplicate-check", "--any-label", "--strict",
        ])
        assert code == 1

    def test_info_reports_model_cost(self, capsys):
        assert main(["info", "--set", "model.preset=nano", "--iterations", "2"]) == 0
        out = capsys.readouterr().out
        for field in ("parameters", "MACs", "latency", "throughput", "realtime factor"):
            assert field in out

    def test_info_respects_an_overridden_resolution(self, capsys):
        main(["info", "--set", "model.preset=nano",
              "--set", "data.image_size=64", "--iterations", "2"])
        assert "64x64" in capsys.readouterr().out
