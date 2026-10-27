"""Shared state reads tolerate replacement gaps, but preserve real failures."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.controller import storage


def test_missing_state_has_a_bounded_retry_window(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(storage, 'time', SimpleNamespace(sleep=sleeps.append))
    with pytest.raises(FileNotFoundError):
        storage.read_state(tmp_path/'missing.json')
    assert sleeps == [.1] * 9


@pytest.mark.parametrize('error', [json.JSONDecodeError('Invalid', '', 0), PermissionError('Denied')])
def test_invalid_or_inaccessible_state_fails_without_retry(tmp_path, monkeypatch, error):
    reads, sleeps = [], []

    def fail(path):
        reads.append(path)
        raise error

    monkeypatch.setattr(Path, 'read_text', fail)
    monkeypatch.setattr(storage, 'time', SimpleNamespace(sleep=sleeps.append))
    path = tmp_path/'state.json'
    with pytest.raises(type(error)) as caught:
        storage.read_state(path)
    assert caught.value is error
    assert reads == [path] and sleeps == []
