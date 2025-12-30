## 2024-05-23 - [Initial Setup]
**Learning:** Performance optimization often starts with identifying tight loops in critical paths.
**Action:** Use profiling or code analysis to find O(N^2) or repeated heavy calculations.

## 2024-05-23 - [Vectorized Correlation]
**Learning:** Iterative `np.correlate` in Python loops is significantly slower (~3-4x) than batched FFT-based correlation, even for "safe" fallbacks.
**Action:** When implementing "safe" or "robust" fallbacks, ensure vectorization is used where possible (e.g., FFT padding for linear correlation) to maintain performance without sacrificing correctness.
