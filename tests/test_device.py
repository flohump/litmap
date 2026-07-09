"""Device selection: mps -> cuda -> cpu, with an env override.

`device="mps"` was hardcoded, so litmap raised on any machine without Apple
Silicon. These tests inject a fake torch module rather than probing the real
one, so they assert the selection logic on every host.
"""
import sys
import types

import pytest

from litmap.embedder import _detect_device


def _fake_torch(*, mps: bool, cuda: bool) -> types.ModuleType:
    mod = types.ModuleType("torch")
    backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    mod.backends = backends
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    return mod


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("LITMAP_DEVICE", raising=False)


@pytest.mark.parametrize(
    "mps,cuda,expected",
    [
        (True, True, "mps"),    # mps wins when both available
        (True, False, "mps"),
        (False, True, "cuda"),
        (False, False, "cpu"),
    ],
)
def test_detect_device_precedence(monkeypatch, no_env, mps, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=mps, cuda=cuda))
    assert _detect_device() == expected


def test_detect_device_env_override_wins(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(mps=True, cuda=True))
    monkeypatch.setenv("LITMAP_DEVICE", "cpu")
    assert _detect_device() == "cpu"


def test_detect_device_falls_back_to_cpu_without_torch(monkeypatch, no_env):
    """A broken/absent torch must not crash device selection."""
    # `import torch` with None in sys.modules raises ImportError.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _detect_device() == "cpu"
