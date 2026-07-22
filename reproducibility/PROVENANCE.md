# Numerical provenance

The historical canonical BW cache and a fresh execution of the exported source
code are not byte-identical. This discrepancy predates the repository refactor.

- Historical cache SHA-256: `dfc367293485b45e2e0f65b8589183a45732e0f70995d91c95eba9ebb57c86fe`
- Fresh source/refactor output SHA-256: `5551f787be3e4c3fc0f7720c36769cb9676a00cd4a0fea211fbaba0623d2fe52`
- Maximum absolute distance difference: `0.353935138561752`
- Historical transition-peak/post mean ratio: `3.055966057792106`
- Fresh transition-peak/post mean ratio: `3.0548107394450716`

A fresh run of the untouched source analysis and the refactored distance-only
analysis produced exactly the same 50 midpoint and distance values. Therefore,
the difference is attributable to an earlier unrecorded cache-generating state,
not to the refactor. The reported transition conclusion is stable, but numerical
claims should use the fresh reproducible output.

Audit two files with:

```powershell
python reproducibility/Grokking/Validation/02_verify_bw_provenance.py `
  --cached path/to/bw_geodesic_results.json `
  --rerun path/to/resolvent_bw_results.json `
  --output path/to/provenance_report.json
```
