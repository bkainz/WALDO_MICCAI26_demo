# WALDO results

Genuine full-run, per-image outputs that back the paper tables. Recompute the headline
metrics with:

```bash
python scripts/read_results.py --dataset all
```

See the top-level [README](../README.md) for the full results tables and
[DISCLAIMER.md](../DISCLAIMER.md) for provenance.

## NOVA brain MRI — `nova/` (seed 42)

`n` = evaluation sample size (906 = full NOVA test set; 50 = API-cost-limited run).

| File | Model | Method | n | mAP@30 |
|---|---|---|:--:|---|
| `nova_zeroshot_qwen25_72b.json`     | Qwen2.5-VL-72B     | Zero-shot | 906 | 36.4% |
| `nova_waldo_qwen25_72b.json`        | Qwen2.5-VL-72B     | WALDO     | 906 | **43.5%** |
| `nova_zeroshot_qwen3_32b.json`      | Qwen3-VL-32B       | Zero-shot | 906 | 20.4% |
| `nova_waldo_qwen3_32b.json`         | Qwen3-VL-32B       | WALDO     | 50  | 32.0% |
| `nova_zeroshot_qwen3_235b.json`     | Qwen3-VL-235B (MoE)| Zero-shot | 906 | 36.3% |
| `nova_waldo_qwen3_235b.json`        | Qwen3-VL-235B (MoE)| WALDO     | 906 | 31.8% |
| `nova_zeroshot_gemini20flash.json`  | Gemini-2.0-Flash   | Zero-shot | 906 | 18.1% |
| `nova_waldo_gemini20flash.json`     | Gemini-2.0-Flash   | WALDO     | 50  | 38.0% |

Each file holds `config`, `metrics` (mAP@30/50, avg IoU, std-err, 95% bootstrap CI) and
`detailed_results` (per-image `filename`, `iou`, `hit_30`, `hit_50`); all recompute to the
mAP@30 shown. **GPT-4o NOVA** rows (zero-shot 19.0 / WALDO 32.0) are the mean of multiple
n=50 API runs and are not shipped as single per-image files.

The 906-sample runs recorded `n_refs: 50` (a 50-image healthy reference pool, within the
paper's stated N=30–50). The paper used 30 IXI healthy brain MRIs as NOVA references whereas
the demo loader uses NOVA "no-finding" scans — see [DISCLAIMER.md](../DISCLAIMER.md).

## VinDr-CXR — `cxr/` (*n*=949 with ≥1 annotated finding)

mAP@30 below is **recomputed from the shipped per-image JSONs** (`read_results.py`); it
matches the paper to within run-to-run variance (≤0.5%).

| File | Model | Method | mAP@30 |
|---|---|---|---|
| `cxr_zeroshot_gpt4o.json`        | GPT-4o         | Zero-shot | 3.4%  |
| `cxr_waldo_gpt4o.json`           | GPT-4o         | WALDO     | 10.9% |
| `cxr_zeroshot_qwen25_72b.json`   | Qwen2.5-72B    | Zero-shot | 18.3% |
| `cxr_waldo_qwen25_72b.json`      | Qwen2.5-72B    | WALDO     | 22.3% |
| `cxr_zeroshot_qwen3_32b.json`    | Qwen3-32B      | Zero-shot | 12.8% |
| `cxr_waldo_qwen3_32b.json`       | Qwen3-32B      | WALDO     | 34.1% |

CXR files additionally include `pred_boxes`, `gt_boxes`, and the raw VLM `responses` per image.
