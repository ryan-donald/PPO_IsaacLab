import torch

from ryan_ppo.normalization import ObsNormalization


def test_welford_matches_batch_statistics():
    # streaming mean/var over several batches must match the statistics
    # computed over the concatenated data directly.
    torch.manual_seed(0)
    norm = ObsNormalization(state_dim=6)

    batches = [torch.randn(128, 6) * (i + 1) + i for i in range(5)]
    for batch in batches:
        norm.update(batch)

    all_data = torch.cat(batches)
    assert torch.allclose(norm.mean, all_data.mean(dim=0), atol=1e-4)
    assert torch.allclose(norm.var, all_data.var(dim=0, unbiased=False), atol=1e-3)


def test_forward_normalizes():
    # after updating on data, the normalized data should be ~zero mean, unit std.
    torch.manual_seed(0)
    norm = ObsNormalization(state_dim=3)

    data = torch.randn(1024, 3) * 5.0 + 2.0
    norm.update(data)
    out = norm(data)

    assert torch.allclose(out.mean(dim=0), torch.zeros(3), atol=1e-2)
    assert torch.allclose(out.std(dim=0), torch.ones(3), atol=1e-2)
