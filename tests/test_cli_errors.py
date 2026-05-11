"""CLI error handling regression tests."""

import json
import os
import subprocess
import sys


COMMANDS = [
    (
        ["store", "hello", "cli", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["recall", "hello", "not-an-int"],
        "top_k must be an integer",
    ),
    (
        ["update", "missing-id", "new content", "not-a-float"],
        "importance must be a number",
    ),
    (
        ["import", "missing-file.json"],
        "Import file not found",
    ),
]


def run_cli(args, tmp_path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
    return subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_import_hindsight_errors_return_nonzero_exit(tmp_path):
    missing_file = tmp_path / "missing-hindsight-export.json"

    result = run_cli(["import-hindsight", str(missing_file)], tmp_path)

    assert result.returncode != 0
    assert "Traceback" not in result.stdout
    assert "Traceback" not in result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider"] == "hindsight"
    assert payload["errors"]
    assert "No such file or directory" in payload["errors"][0]


def test_invalid_cli_input_reports_error_without_traceback(tmp_path):
    for args, expected_error in COMMANDS:
        result = run_cli(args, tmp_path)

        assert result.returncode != 0, args
        assert expected_error in result.stderr, result.stderr
        assert "Traceback" not in result.stderr


def test_import_non_object_json_reports_error_without_traceback(tmp_path):
    for payload in ("[]", '"not an export"'):
        bad_export = tmp_path / "not-an-export.json"
        bad_export.write_text(payload, encoding="utf-8")

        result = run_cli(["import", str(bad_export)], tmp_path)

        assert result.returncode != 0
        assert "Import file must contain a Mnemosyne export object" in result.stderr
        assert "Traceback" not in result.stderr


def test_import_malformed_json_reports_error_without_traceback(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")

    result = run_cli(["import", str(bad_json)], tmp_path)

    assert result.returncode != 0
    assert "Invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr


def test_bank_cli_list_create_delete_uses_configured_data_dir(tmp_path):
    result = run_cli(["bank", "list"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "default" in result.stdout
    assert "Traceback" not in result.stderr

    result = run_cli(["bank", "create", "project_a"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Created bank: project_a" in result.stdout
    assert "Traceback" not in result.stderr

    result = run_cli(["bank", "list"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "project_a" in result.stdout

    result = run_cli(["bank", "delete", "project_a"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Deleted bank: project_a" in result.stdout
    assert "Traceback" not in result.stderr


def test_bank_cli_validation_errors_are_user_facing(tmp_path):
    cases = [
        (["bank", "create", "bad/name"], "Invalid bank name", 2),
        (["bank", "create"], "Usage: mnemosyne bank create <name>", 2),
        (["bank", "delete"], "Usage: mnemosyne bank delete <name>", 2),
        (["bank", "nope"], "Unknown bank command: nope", 2),
        (["bank", "delete", "missing_bank"], "Bank not found: missing_bank", 1),
    ]

    for args, expected_error, expected_returncode in cases:
        result = run_cli(args, tmp_path)

        assert result.returncode == expected_returncode, args
        assert result.stdout == ""
        assert expected_error in result.stderr
        assert "Traceback" not in result.stderr
