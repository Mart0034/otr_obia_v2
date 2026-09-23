"""Tests for cli.py: settings can come from CONFIG defaults, an optional
JSON file, or command-line flags, with flags always winning."""
import json

from cli import build_arg_parser, resolve_config


def test_cli_flags_override_base_config():
    parser = build_arg_parser("test")
    args = parser.parse_args(["--n-estimators", "10", "--min-overlap-ratio", "0.5"])
    base = {"n_estimators": 400, "min_overlap_ratio": 0.3, "other": "kept"}

    cfg = resolve_config(base, args)

    assert cfg["n_estimators"] == 10
    assert cfg["min_overlap_ratio"] == 0.5
    assert cfg["other"] == "kept"  # untouched settings stay as-is


def test_config_file_overrides_base_but_cli_flags_win(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"n_estimators": 50, "output_dir": "from_file"}))

    parser = build_arg_parser("test")
    args = parser.parse_args(["--config", str(config_path), "--n-estimators", "99"])
    base = {"n_estimators": 400, "output_dir": "default"}

    cfg = resolve_config(base, args)

    assert cfg["output_dir"] == "from_file"  # came from the config file
    assert cfg["n_estimators"] == 99  # CLI flag wins over the config file


def test_no_overrides_keeps_base_config_unchanged():
    parser = build_arg_parser("test")
    args = parser.parse_args([])
    base = {"n_estimators": 400, "output_dir": "default"}

    cfg = resolve_config(base, args)

    assert cfg == base


def test_classification_threshold_flag_overrides_default():
    parser = build_arg_parser("test")
    args = parser.parse_args(["--classification-threshold", "0.8"])
    base = {"classification_threshold": 0.5}

    cfg = resolve_config(base, args)

    assert cfg["classification_threshold"] == 0.8
