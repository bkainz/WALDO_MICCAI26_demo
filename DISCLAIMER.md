# Disclaimer

## AI-assisted distillation

This repository is a **clean, self-contained reference implementation** of the WALDO method
described in the paper

> *Wasserstein-Aligned Localisation for VLM-Based Distributional OOD Detection in Medical
> Imaging*, B. Kainz, J. P. Mueller, M. M. G. Baugh, C. I. Bercea, MICCAI 2026.

The code here was **distilled by Claude (Anthropic) and Codex (OpenAI) coding agents from a
larger internal experimentation codebase**. That original codebase comprised many one-off
scripts run across a multi-day GPU-cluster job; this demo consolidates the method into a
small, readable, runnable package.

### What this means

- **The algorithm and hyper-parameters here faithfully follow the paper**: DINOv3-ViT-B/16
  patch features; entropy-weighted, squared Sliced Wasserstein distance (*M*=100 projections);
  Goldilocks-zone selection (30th–70th percentile, α=0.3); Determinantal Point Process
  diversity; two-stage differential prompting (*K*=5 references, 3 in Stage 1 + 2 in Stage 2);
  confidence-weighted NMS at IoU 0.5 with `c·exp(−λ·SW)`, λ=0.1; VLM sampling at
  temperature 0.7, top-p 0.95.
- **This is not the exact orchestration script** used to generate the published tables. Some
  engineering details (image I/O, batching, client retry logic, the precise sequence of
  cluster jobs) differ. Running this demo end-to-end will not bit-for-bit reproduce the paper
  numbers without the full datasets, the gated DINOv3 backbone, and the same VLM endpoints.
- **The numbers in `README.md` are the published paper results.** The JSON files in
  [`results/`](results/) are the **genuine full-run, per-image outputs** that back them
  (NOVA *n*=906; VinDr-CXR *n*=949). They are provided so the reported metrics can be
  recomputed directly (`python scripts/read_results.py --dataset all`).
- For NOVA, the reported GPT-4o figures are based on repeated (n=50) API evaluations and are reported as the mean across these runs. Per-image JSON files are available for all other NOVA rows where they were generated and retained. For GPT-4o, the retained evidence is the repeated-run aggregate rather than a single per-image JSON file. The original full-cohort GPT-4o JSON file covering all (n=906) NOVA cases could not be included because it was lost when the temporary cluster workspace exceeded its storage-retention window. The file could not be recovered from the cluster. A full rerun over all 906 cases would incur substantial additional API token and energy costs; given that the current estimate is already based on repeated (n=50) runs. 

### Known, intentional differences (full transparency)

So that nothing looks hidden, these are the places where the demo deliberately differs from
the paper's exact experimental setup:

- **Differential prompt.** The paper quotes the core differential instruction verbatim
  ("Compare the patient scan (left) to the healthy reference (right) ... normalised to
  [0, 1000]"). The demo embeds that exact sentence but wraps it with a small JSON output
  contract (and FIRST/SECOND image wording, since the query and reference are sent as two
  separate images rather than a side-by-side composite) so responses can be parsed robustly.
- **NOVA healthy references.** The paper used 30 IXI healthy brain MRIs as the NOVA
  reference pool. For convenience (no extra dataset download) the demo loader uses NOVA
  "no-finding" scans as references. The shipped NOVA full-run JSONs record `n_refs: 50`
  (within the paper's stated N=30–50 range); that is the genuine pool size of those runs and
  has been left unmodified.
- **Two-stage aggregation.** The paper's aggregation equation is a single weighted NMS over
  all K references; the demo runs Stage 1 (differential) and Stage 2 (refinement) as separate
  weighted-NMS passes where Stage 2 replaces Stage 1 on confirmation, and falls back to the
  Stage-1 candidates if Stage 2 returns nothing. This is a behavioural elaboration of the
  paper's prose, not a contradiction of a governing equation.
- **Reproduction variance.** Recomputing the shipped JSONs with `read_results.py` matches the
  paper to within run-to-run VLM variance: all WALDO rows ≤0.1 pp; the largest baseline gap is
  Qwen2.5-72B CXR zero-shot (18.3% recomputed vs 18.7% reported) and GPT-4o CXR zero-shot
  (3.4% vs 3.3%).

If you find any other discrepancy between this code and the paper, please open an issue — we
want the released code to match the published method exactly.

## Intended use

WALDO targets **research and triage / attention-guidance** settings for rare-pathology
localisation where no labelled training data is available. At ~43.5% mAP@30 on NOVA, and with
small lesions (<5% image area) frequently missed, it is **not** a diagnostic device and must
not be used for primary clinical diagnosis. Healthy reference pools should be curated with
clinical expertise.

## License

The code is released under the MIT License.
