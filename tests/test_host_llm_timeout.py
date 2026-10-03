"""Host-LLM per-attempt timeout: env-configurable, default clears reasoning-aux latency."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _timeout(env_value=None):
    env = {k: v for k, v in os.environ.items() if k != "MNEMOSYNE_HOST_LLM_TIMEOUT"}
    if env_value is not None:
        env["MNEMOSYNE_HOST_LLM_TIMEOUT"] = env_value
    out = subprocess.check_output(
        [sys.executable, "-c", "from mnemosyne.core import local_llm; print(local_llm.HOST_LLM_TIMEOUT)"],
        cwd=ROOT, env=env, text=True,
    )
    return float(out.strip())


def test_default_exceeds_old_15s_cap():
    assert _timeout() == 60.0


def test_env_override():
    assert _timeout("90") == 90.0


def test_invalid_values_fall_back_to_default():
    for bad in ("abc", "0", "-5"):
        assert _timeout(bad) == 60.0, bad
