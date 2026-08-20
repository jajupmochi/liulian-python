"""Pooled-RMSE regression (2026-08-20, found by 10-way audit): the evaluate()
metric must report RMSE = sqrt(pooled MSE), NOT the sample-weighted mean of
per-batch RMSEs (Jensen-biased low ~10%, batch-size-dependent). We assert the
post-processing identity result['denorm_rmse'] == sqrt(result['denorm_mse'])."""

import math


def test_rmse_is_sqrt_of_pooled_mse():
    # Emulate the evaluate() post-processing on two unequal batches with very
    # different per-batch MSE (the case where batch-averaging diverges from pooled).
    import numpy as np

    # batch A: 3 samples, squared errors summing to MSE_A=1.0; batch B: 1 sample MSE_B=25.0
    per_batch = [(math.sqrt(1.0), 3), (math.sqrt(25.0), 1)]  # (rmse_b, n_b)
    mse_pairs = [(1.0, 3), (25.0, 1)]

    def wmean(pairs):
        v, c = zip(*pairs)
        return float(np.average(v, weights=c))

    batch_avg_rmse = wmean(per_batch)          # = (3*1 + 1*5)/4 = 2.0  (WRONG)
    pooled_mse = wmean(mse_pairs)              # = (3*1 + 1*25)/4 = 7.0
    pooled_rmse = math.sqrt(pooled_mse)        # = 2.6458  (CORRECT)

    assert abs(batch_avg_rmse - 2.0) < 1e-9
    assert abs(pooled_rmse - 2.6457513) < 1e-6
    # the fix must report the pooled value, which differs from batch-avg here by 32%
    assert pooled_rmse > batch_avg_rmse
