from unittest.mock import Mock

import pytest
import torch

from ryan_ppo.utils import PhaseTimer, get_device


def test_explicit_device_does_not_probe_cuda(monkeypatch):
    probe = Mock(side_effect=AssertionError("CUDA should not be probed"))
    monkeypatch.setattr(torch.cuda, "is_available", probe)
    assert get_device("cpu") == torch.device("cpu")
    assert get_device("cuda:1") == torch.device("cuda:1")
    probe.assert_not_called()


def test_cpu_phase_timer(monkeypatch):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr("ryan_ppo.utils.time.perf_counter", lambda: next(clock))
    timer = PhaseTimer(torch.device("cpu"))
    timer.start()
    timer.stop()
    assert timer.seconds() == pytest.approx(0.25)


def test_cuda_timer_defers_synchronization_without_using_gpu(monkeypatch):
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 250.0
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=[start, end]))
    stream = Mock()
    get_stream = Mock(return_value=stream)
    monkeypatch.setattr(torch.cuda, "current_stream", get_stream)
    timer = PhaseTimer(torch.device("cuda:1"))
    timer.start()
    timer.stop()
    start.record.assert_called_once_with(stream)
    end.record.assert_called_once_with(stream)
    get_stream.assert_called_with(torch.device("cuda:1"))
    end.synchronize.assert_not_called()
    assert timer.seconds() == pytest.approx(0.25)
    end.synchronize.assert_called_once()
