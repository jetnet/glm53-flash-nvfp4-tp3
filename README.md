# GLM-5.3-Flash NVFP4 on 3× DGX Spark — TP=3, 512K context, 35 tok/s

Serve [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)
(320B total / 18B active, hybrid KDA + NoPE sparse-MLA) across **three** DGX Spark
(GB10, SM121) nodes at tensor-parallel 3.

Measured here: **35.2 tok/s** decode with native MTP-4, TTFT 0.26 s, needle retrieval
verified at **471,813 tokens**, Xid 0 on all ranks, one shared checkpoint over NFS.

Nothing in GLM-5.3's geometry divides by three, so this needs head padding, four upstream
overlay files, **and one local kernel fallback that upstream does not have** — without it
every request past ~25K tokens kills the engine on GB10. See [docs/GOTCHAS.md](docs/GOTCHAS.md).

## What this repo is

The delta on top of [FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks](https://github.com/FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks)
(MIT, which in turn rebuilds [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark)'s
SM121 patch ladder). You need that repo for the base TP=3 overlay files and v8 image
recipe; this repo carries the DFlash2 v12 overlay needed to reproduce the PR result.
This repo adds:

| File | Purpose |
| --- | --- |
| `scripts/serve.sh` | Rank-driven launcher: shared-checkpoint (NFS or local), `TRANSPORT=ib\|mesh\|socket` |
| `scripts/build-indexer-overlay.py` | **The GB10 fix**: sparse-indexer top-k fallback past ~25K tokens |
| `scripts/pad-config.py` | On-disk `config.json` pad: 66 heads / **2112** MoE I (upstream writes 2049 — mismatch) |
| `scripts/fetch-weights.sh` | Weights + the `chat_template.jinja` that weight-only downloaders skip |
| `scripts/ship-image.sh` | `docker save \| ssh docker load` to peers |
| `scripts/pad-dflash2-drafter.py` | **DFlash2 at TP=3**: head-pads the drafter 32/8 -> 48/12 |
| `scripts/extract-nvrtc-header.sh` | Extracts `nvrtc.h` from the image's NVIDIA NVRTC wheel into the local overlay |
| `docker/Dockerfile.glm53-sm121-v12-dflash2` | Reproducible DFlash2 v12 image build |
| `docker/dflash2-overlay/` | DFlash2 model glue, aux capture, registry/select, and KV drafter-group patches |
| `tests/` | tok/s, long-context needle, CJK-leak probe |

## Requirements

- 3× DGX Spark (GB10 / SM121), ~120 GiB unified memory each, driver 580+, Docker.
- ~185 GiB free for the checkpoint (one copy is enough — see below) + ~45 GiB for the image.
- Any L2 network shared by all three nodes (1 GbE management is fine) **plus** RDMA links.
  Tensor traffic goes over RDMA; the shared L2 only carries rendezvous.
- `docker` group membership on **every** node (easy to miss on a node added later).

## Install

```bash
# 0. clone this + upstream (upstream provides the Dockerfile and the 4 overlay files)
git clone https://github.com/<you>/glm53-flash-nvfp4-tp3 ~/src/glm53-tp3
git clone https://github.com/FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks ~/src/glm53-upstream

# 1. weights + chat template (rank 0 only if you share them; every node otherwise)
~/src/glm53-tp3/scripts/fetch-weights.sh /path/to/glm-5.3-flash-nvfp4

# 2. build the base image on one node (~50 min: 20.7 GB base pull + a FlashInfer nightly)
cd ~/src/glm53-upstream && docker build -f docker/Dockerfile.sm121-v8 -t glm53:sm121-v8 docker/
~/src/glm53-tp3/scripts/ship-image.sh glm53:sm121-v8 <peer1> <peer2>

# 3. build the GB10 indexer overlay on every node
~/src/glm53-tp3/scripts/build-indexer-overlay.py --image glm53:sm121-v8 --out ~/src/glm53-overlay

# 4. pad the checkpoint config once (writable copy; backs up config.json.orig)
~/src/glm53-tp3/scripts/pad-config.py /path/to/glm-5.3-flash-nvfp4/config.json --tp 3

# 5. configure and copy to every node
cp cluster.env.example cluster.env && $EDITOR cluster.env
```

## Run

Workers first, rank 0 last. Tear all three down before relaunching any.

```bash
# on each node, in this order: rank 2, rank 1, rank 0
./scripts/serve.sh up 2
./scripts/serve.sh up 1
./scripts/serve.sh up 0     # this one exposes the API

./scripts/serve.sh down      # on all three
```

Load takes ~8 min (120 shards) plus ~3 min of engine warmup. Then:

```bash
curl http://<rank0>:8045/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"glm-5.3-flash",
  "messages":[{"role":"user","content":"hi"}],
  "max_tokens":600,
  "chat_template_kwargs":{"reasoning_effort":"low"}}'
```

`reasoning` and `content` come back as separate fields. Give `max_tokens` room: this model
**always** thinks, and there is no way to turn that off (see gotchas).

## DFlash2 speculative decoding (optional)

47.2 tok/s thinking-on, **+28%** over MTP-3 on this fleet, with the pool at 1,366,425
tokens (2.61x). Build the DFlash2 v12 image from this branch, then prepare the two
runtime overlays that are intentionally local machine state:

```bash
# image v12 = sm121-v8 + DFlash2 registry/model glue + GLM aux capture + drafter KV grouping
docker build -f docker/Dockerfile.glm53-sm121-v12-dflash2 -t glm53:sm121-v12-dflash2 .

# FlashInfer JIT needs this header under /usr/local/cuda/include; extract it from
# the NVIDIA CUDA NVRTC Python wheel installed in the image, do not commit the header.
scripts/extract-nvrtc-header.sh glm53:sm121-v12-dflash2 ~/src/glm53-overlay

# serve.sh bind-mounts the TP=3 GLM model overlay; for DFlash2 it must also carry
# the aux-hidden-state capture patch baked into image v12.
cp ~/src/glm53-upstream/overlay/vllm/models/glm5next/nvidia/model.py \
  ~/src/glm53-overlay/glm5next_model_dflash2.py
python3 docker/dflash2-overlay/patch_glm_aux_capture.py \
  --model-file ~/src/glm53-overlay/glm5next_model_dflash2.py

# 32 q / 8 kv heads -> 48 q / 12 kv, zero-padded; verifies numerically before writing
scripts/pad-dflash2-drafter.py ~/glm53-dflash2-draft ~/glm53-dflash2-draft-tp3
```

Then set `IMAGE=glm53:sm121-v12-dflash2`, `SPEC_METHOD=dflash`, and `DFLASH2_DIR`
in `cluster.env` (see `cluster.env.example`). Two traps worth knowing before you try a
different pad:

- `draft_tensor_parallel_size=1` does **not** let the drafter escape sharding.
  `load_dflash_model()` builds the draft under the *target's* parallel config and never
  applies `draft_parallel_config` — the setting is silently ignored.
- 36 q / 9 kv also divides by 3 and also keeps the GQA ratio, but lands the drafter on the
  standalone KV path, where it keeps `block_size=16` and one 512K request wants 33.6 GiB.
  48/12 hits exact page fit at block 2304 and wants 3.2 GiB.

Check `/metrics` after. Acceptance is workload-dependent — with verified aux capture this
lane measured 0.165–0.190 on Russian free prose, 0.181–0.188 on Russian agentic prose,
0.964 on edits, 0.442 on short English code — so there is no universal threshold, and a
low number on free prose is not a broken capture. Verify the aux-hidden-state capture
with a fixed, known-predictable edit/code prompt instead: replay it and compare the
per-position curve against the known-good acceptance 0.91, per-pos 14/13/13/13/13/12/11
of 14. A wrong capture shows up on that prompt as a flat, near-zero curve — it degrades
silently rather than crashing. Full writeup: [results/dflash2-tp3-2026-08-28.md](results/dflash2-tp3-2026-08-28.md).

## Verify

```bash
# decode tok/s + TTFT
tests/measure_tps.py --url http://<rank0>:8045/v1 --runs 3 --max-tokens 300

# long context: exact needle retrieval (needs the indexer overlay)
tests/long_context_needle.py --base-url http://<rank0>:8045/v1 \
    --target-tokens 238000 --max-tokens 600 --thinking on --reasoning-effort low \
    --output results/needle-238k.json

# CJK leakage in non-English output, per sampling configuration
tests/cjk_leak_probe.py --url http://<rank0>:8045/v1/chat/completions
```

## Sizing the KV pool

Concurrency is `kv_pool_tokens / max_model_len`. Measured bytes per token per rank:

| Speculation | B/token/rank |
| --- | ---: |
| off | 6,962 |
| MTP-4 | 8,793 |

`KV_CACHE_MEMORY ≈ max_model_len × target_concurrency × bytes_per_token`. Example, the
default in `cluster.env.example`: 524,288 × 2.5 × 8,793 ≈ **11.5 GB** → measured pool
1,505,849 tokens = 2.87×. Leaving it unset gives ~18× and wastes ~27 GiB per node.

## Numbers

| Profile | Weights/rank | KV pin | Pool | Conc. | TTFT | Decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no spec, 262K | 62.14 GiB | 4.38 GiB | 675,219 | 2.58× | 0.16 s | 18.4 tok/s |
| MTP-4, 262K | 63.64 GiB | 4.38 GiB | 534,635 | 2.04× | 0.18 s | 32.0 tok/s |
| **MTP-4, 512K** | 63.79 GiB | 10.73 GiB | 1,505,849 | 2.87× | 0.26 s | **35.2 tok/s** |

Long context (exact needle retrieval, all PASS): 64K → 44.5 s, 131K → 73.8 s,
238K → 126.2 s, **471,813 → 274.6 s** (~1,800 tok/s prefill).

GPU clock: decode is fabric-bound, prefill is not. 1500 → 2100 MHz gives **+4% decode but
−26% prefill wall clock**, +1 °C. Raise the clock if your prompts are long.

Full detail: [docs/RESULTS.md](docs/RESULTS.md).

## One checkpoint, not three

185 GiB × 3 local copies is unnecessary. Here rank 0 owns the checkpoint and ranks 1–2 read
it read-only over NFSv4.2 on the RDMA fabric (`nconnect=8`, 1 MiB I/O). Load time was
462 s for all three ranks, i.e. NFS was not the bottleneck — ranks finished within seconds
of each other. Point `MODEL_DIR` at the same path everywhere and keep the pad step on the
writable node.

Caveat: at ~64 GiB of weights per rank there is still page-cache headroom. A heavier
checkpoint (e.g. the ~328 GB official FP8 → ~102 GiB/rank) would leave 8–17 GiB for engine,
activations and KV, and NFS loading in that regime is a bad idea.

## Credits

- [FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks](https://github.com/FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks) — TP=3 pad set, overlay files, Dockerfile (MIT)
- [tonyd2wild](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-2x-DGX-Spark) — SM121 NoPE-MLA patch ladder v1–v8
- [LibertAIDAI](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) — the NVFP4 quant; [Z.ai](https://huggingface.co/zai-org/GLM-5.3-Flash) — the model
