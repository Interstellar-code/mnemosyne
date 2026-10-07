import os
import subprocess
import sys


def _sleep_age(env):
    code = "from mnemosyne.core import beam; print(beam.SLEEP_AGE_HOURS)"
    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def test_sleep_age_defaults_to_half_ttl():
    assert _sleep_age({"MNEMOSYNE_WM_TTL_HOURS": "168"}) == 84


def test_sleep_age_decoupled_from_raised_ttl():
    assert _sleep_age({"MNEMOSYNE_WM_TTL_HOURS": "87600", "MNEMOSYNE_SLEEP_AGE_HOURS": "84"}) == 84
