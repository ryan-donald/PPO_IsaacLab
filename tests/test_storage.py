"""RolloutStorage.flatten: shapes, row alignment, and the keyword-only guard."""

import pytest
import torch

from ryan_ppo.storage import RolloutBatch, RolloutStorage

STEPS, ENVS, STATE_DIM, ACTION_DIM, TERMS = 4, 3, 5, 2, 2
N = STEPS * ENVS


def marker(steps, envs, *trailing):
    # every (step, env) slot gets a unique value, so a misaligned flatten shows up
    # as rows that disagree about which (step, env) they came from.
    tag = torch.arange(steps, dtype=torch.float).unsqueeze(1) * 100 + torch.arange(
        envs, dtype=torch.float
    )
    return (
        tag.reshape(steps, envs, *([1] * len(trailing)))
        .expand(steps, envs, *trailing)
        .clone()
        if trailing
        else tag.clone()
    )


def filled_storage():
    storage = RolloutStorage(
        STEPS, ENVS, STATE_DIM, ACTION_DIM, TERMS, torch.device("cpu")
    )
    for step in range(STEPS):
        storage.add(
            step,
            state=marker(STEPS, ENVS, STATE_DIM)[step],
            action=marker(STEPS, ENVS, ACTION_DIM)[step],
            log_prob=marker(STEPS, ENVS)[step],
            reward=torch.zeros(ENVS),
            done=torch.zeros(ENVS),
            trunc=torch.zeros(ENVS),
            term_reward=torch.zeros(ENVS, TERMS),
            mu=marker(STEPS, ENVS, ACTION_DIM)[step],
        )
    return storage


def test_flatten_shapes():
    storage = filled_storage()
    std = torch.rand(ACTION_DIM)

    batch = storage.flatten(
        returns=marker(STEPS, ENVS),
        advantages=marker(STEPS, ENVS),
        values_old=marker(STEPS, ENVS),
        std_old=std,
    )

    assert isinstance(batch, RolloutBatch)
    assert len(batch) == N
    assert batch.states.shape == (N, STATE_DIM)
    assert batch.actions.shape == (N, ACTION_DIM)
    assert batch.mus_old.shape == (N, ACTION_DIM)
    for flat in (
        batch.log_probs_old,
        batch.returns,
        batch.advantages,
        batch.values_old,
    ):
        assert flat.shape == (N,)
    assert batch.std_old is std  # policy-level, passed through unflattened


def test_flatten_keeps_every_field_row_aligned():
    # each slot holds step*100 + env; after flattening, all seven per-sample
    # fields must report the same (step, env) on every row.
    storage = filled_storage()

    batch = storage.flatten(
        returns=marker(STEPS, ENVS),
        advantages=marker(STEPS, ENVS),
        values_old=marker(STEPS, ENVS),
        std_old=torch.rand(ACTION_DIM),
    )

    expected = torch.tensor(
        [s * 100 + e for s in range(STEPS) for e in range(ENVS)], dtype=torch.float
    )
    assert torch.equal(batch.log_probs_old, expected)
    assert torch.equal(batch.returns, expected)
    assert torch.equal(batch.advantages, expected)
    assert torch.equal(batch.values_old, expected)
    for column in range(STATE_DIM):
        assert torch.equal(batch.states[:, column], expected)
    for column in range(ACTION_DIM):
        assert torch.equal(batch.actions[:, column], expected)
        assert torch.equal(batch.mus_old[:, column], expected)


def test_flatten_rejects_positional_args():
    # the three (num_steps, num_envs) tensors are indistinguishable positionally,
    # so passing them by position must not be possible.
    storage = filled_storage()
    r = marker(STEPS, ENVS)

    with pytest.raises(TypeError):
        storage.flatten(r, r, r, torch.rand(ACTION_DIM))
