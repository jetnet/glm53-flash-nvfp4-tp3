#!/usr/bin/env bash
#
# Launch one rank of GLM-5.3-Flash-NVFP4 at TP=3 on a DGX Spark (GB10 / SM121).
# Workers first, rank 0 last. Tear down ALL ranks before relaunching any.
#
#   serve.sh up   [rank]
#   serve.sh down
#   serve.sh args [rank]     # print docker argv, launch nothing
#
# Rank defaults to this node's index in $NODES.
set -euo pipefail

ACTION="${1:-up}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
source "${ENV_FILE:-$ROOT/cluster.env}"

: "${NODES:?set NODES=\"ip0 ip1 ip2\" in cluster.env (rank order; rank 0 serves the API)}"
: "${MODEL_DIR:?set MODEL_DIR}"
: "${UPSTREAM_DIR:?set UPSTREAM_DIR (clone of FlyCockpit/GLM-5.3-Flash-3x-DGX-Sparks)}"

read -r -a NODE_LIST <<<"$NODES"
TP_SIZE="${TP_SIZE:-${#NODE_LIST[@]}}"
MASTER_ADDR="${MASTER_ADDR:-${NODE_LIST[0]}}"

IMAGE="${IMAGE:-glm53:sm121-v8}"
NAME="${NAME:-vllm_glm53}"
MODEL_MNT="${MODEL_MNT:-/models/glm-5.3-flash-nvfp4}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/glm53-vllm-cache}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/src/glm53-overlay}"
PORT="${PORT:-8045}"
MASTER_PORT="${MASTER_PORT:-29531}"
IFACE="${IFACE:?set IFACE to the shared-L2 interface, e.g. eth0}"
TRANSPORT="${TRANSPORT:-ib}"
IB_HCAS="${IB_HCAS:-}"
SERVED_NAME="${SERVED_NAME:-glm-5.3-flash}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MTP_TOKENS="${MTP_TOKENS:-4}"
# Speculation method: mtp (native head) | dflash (DFlash2 block-diffusion drafter) | none
SPEC_METHOD="${SPEC_METHOD:-mtp}"
DFLASH2_DIR="${DFLASH2_DIR:-$HOME/glm53-dflash2-draft}"
DFLASH2_MNT="${DFLASH2_MNT:-/models/dflash2-draft}"
# block_size 8 minus the target's own verified token; 8 drafts a position the model never learned.
DFLASH_TOKENS="${DFLASH_TOKENS:-7}"
# No draft_tensor_parallel_size: load_dflash_model() (v1/worker/gpu/spec_decode/dflash/
# utils.py) builds the draft with replace(vllm_config, ...) and never applies
# speculative_config.draft_parallel_config, so the key is silently ignored on the dflash
# path -- the drafter always shards at TARGET TP. That is why the checkpoint must be
# head-padded instead (scripts/pad-dflash2-drafter.py).
ENABLE_EP="${ENABLE_EP:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
REASONING_PARSER="${REASONING_PARSER:-glm47}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-chat_template.jinja}"

if [ "$ACTION" = "down" ]; then
	docker rm -f "$NAME" 2>/dev/null || true
	echo "removed $NAME on $(hostname -s)"
	exit 0
fi

# Rank: explicit argument, or this node's position in NODES.
NODE_RANK="${2:-${NODE_RANK:-}}"
if [ -z "$NODE_RANK" ]; then
	for i in "${!NODE_LIST[@]}"; do
		if ip -4 -o addr show | grep -qw "${NODE_LIST[$i]}"; then NODE_RANK="$i"; fi
	done
fi
[ -n "$NODE_RANK" ] || { echo "cannot infer rank: pass it, or set NODE_RANK" >&2; exit 2; }
HOST_IP="${NODE_LIST[$NODE_RANK]}"

test -f "$MODEL_DIR/config.json" || { echo "no checkpoint at $MODEL_DIR" >&2; exit 2; }
test -d "$UPSTREAM_DIR/overlay/vllm" || { echo "no overlay tree in $UPSTREAM_DIR" >&2; exit 2; }
mkdir -p "$CACHE_DIR"

# SpeculativeConfig reads the checkpoint, not --hf-overrides, so the file itself must be
# padded. Verify on every rank; a shared read-only mount is padded by whoever owns it.
python3 - "$MODEL_DIR/config.json" "$TP_SIZE" <<-'PY'
	import json, sys
	cfg = json.load(open(sys.argv[1]))["text_config"]
	tp = int(sys.argv[2])
	heads, moe = cfg["num_attention_heads"], cfg["moe_intermediate_size"]
	if heads % tp or moe % tp:
	    sys.exit(f"config.json not padded for TP={tp}: heads={heads} moe_i={moe}; "
	             "run scripts/pad-config.py")
PY

HEADLESS=""
[ "$NODE_RANK" != "0" ] && HEADLESS="--headless"

OPT=()
[ -n "$KV_CACHE_MEMORY" ] && OPT+=(--kv-cache-memory "$KV_CACHE_MEMORY")
case "$SPEC_METHOD" in
mtp)
	[ "$MTP_TOKENS" != "0" ] && OPT+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
	;;
dflash)
	test -r "$DFLASH2_DIR/config.json" || { echo "no readable DFlash2 drafter config at $DFLASH2_DIR/config.json" >&2; exit 2; }
	# A present 0644 config is not a working drafter: root-owned 0600 *.safetensors only
	# fails 9.5 min into each load, as FileNotFoundError, on root-squashed NFS workers.
	shopt -s nullglob
	draft_ts=("$DFLASH2_DIR"/*.safetensors)
	shopt -u nullglob
	[ "${#draft_ts[@]}" -gt 0 ] && [ -r "${draft_ts[0]}" ] || { echo "no readable *.safetensors in $DFLASH2_DIR (readable on rank 0 but not on root-squashed NFS workers)" >&2; exit 2; }
	# Verify the checkpoint IS the proven TP=3 pad, not merely present: unpadded 32/8
	# dies at load, the tempting 36/9 pad boots then demands ~20 GiB per 512K request.
	python3 - "$DFLASH2_DIR/config.json" <<-'PY'
		import json, sys
		cfg = json.load(open(sys.argv[1]))
		q, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
		if (q, kv) != (48, 12):
		    sys.exit(f"drafter is not the TP=3 pad 48/12 (got {q}q/{kv}kv); run scripts/pad-dflash2-drafter.py")
	PY
	# A missing/empty/stale nvrtc.h defers the failure to FlashInfer JIT warmup, after the
	# target checkpoint has loaded. extract-nvrtc-header.sh installs it with the same
	# checks; this rejects stale or hand-made state.
	test -s "$OVERLAY_DIR/nvrtc.h" || { echo "SPEC_METHOD=dflash needs a non-empty $OVERLAY_DIR/nvrtc.h (scripts/extract-nvrtc-header.sh)" >&2; exit 2; }
	grep -q nvrtcResult "$OVERLAY_DIR/nvrtc.h" || { echo "$OVERLAY_DIR/nvrtc.h does not look like the CUDA NVRTC header; re-extract it" >&2; exit 2; }
	OPT+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH2_MNT}\",\"num_speculative_tokens\":${DFLASH_TOKENS}}")
	;;
none) ;;
*) echo "SPEC_METHOD must be mtp|dflash|none" >&2; exit 2 ;;
esac
[ "$ENABLE_EP" = "1" ] && OPT+=(--enable-expert-parallel)
[ "$ENFORCE_EAGER" = "1" ] && OPT+=(--enforce-eager)

# 64 heads and MoE 2048 do not divide by 3; VllmConfig validates before model code runs.
if [ $((64 % TP_SIZE)) -ne 0 ]; then
	HEADS=$(((64 + TP_SIZE - 1) / TP_SIZE * TP_SIZE))
	MOE_I="${MOE_I:-2112}"   # GEMM-friendly (multiple of 64); see docs/GOTCHAS.md
	PADS="{\"num_attention_heads\":$HEADS,\"num_key_value_heads\":$HEADS,\"linear_num_heads\":$HEADS,\"moe_intermediate_size\":$MOE_I"
	OPT+=(--hf-overrides "$PADS,\"text_config\":$PADS}}")
fi
# Vision tower is 16 heads: data-parallel the encoder instead of sharding it.
[ $((16 % TP_SIZE)) -ne 0 ] && OPT+=(--mm-encoder-tp-mode "${MM_ENCODER_TP_MODE:-data}")

# enable_thinking cannot silence this model; false only makes the parser merge
# reasoning into content. Keep it true and steer verbosity with reasoning_effort.
KWARGS='{"enable_thinking":true'
[ -n "$REASONING_EFFORT" ] && KWARGS="$KWARGS,\"reasoning_effort\":\"$REASONING_EFFORT\""
OPT+=(--default-chat-template-kwargs "$KWARGS}")

if [ -n "$CHAT_TEMPLATE" ]; then
	test -f "$MODEL_DIR/$CHAT_TEMPLATE" || { echo "missing $MODEL_DIR/$CHAT_TEMPLATE (see scripts/fetch-weights.sh)" >&2; exit 2; }
	OPT+=(--chat-template "$MODEL_MNT/$CHAT_TEMPLATE")
fi

NCCL_ENV=(
	-e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0
	-e NCCL_IB_MERGE_NICS=0 -e NCCL_IGNORE_CPU_AFFINITY=1
	-e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
	-e GLOO_SOCKET_IFNAME="$IFACE" -e TP_SOCKET_IFNAME="$IFACE" -e MN_IF_NAME="$IFACE"
)
PLUGIN=()
case "$TRANSPORT" in
ib)
	: "${IB_HCAS:?set IB_HCAS, e.g. rocep1s0f0,rocep1s0f1 (ls /sys/class/infiniband)}"
	# shellcheck disable=SC2054  # commas belong to the NCCL_IB_HCA value
	NCCL_ENV+=(
		-e NCCL_NET=IB -e NCCL_IB_DISABLE=0
		-e NCCL_IB_HCA="$IB_HCAS"
		-e NCCL_IB_SUBNET_AWARE_ROUTING=1
		-e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET
		-e NCCL_NET_PLUGIN=none -e NCCL_SOCKET_IFNAME="$IFACE"
	)
	;;
mesh)
	MESH_DIR="${MESH_DIR:-$ROOT/nccl-mesh}"
	test -f "$MESH_DIR/libnccl-net-mesh.so" || test -f "$MESH_DIR/libnccl-net.so"
	PLUGIN=(-v "$MESH_DIR:/opt/nccl-mesh:ro")
	NCCL_ENV+=(
		-e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e NCCL_NET_PLUGIN=mesh
		-e NCCL_SOCKET_IFNAME="=${IFACE}" -e NCCL_ALGO=Ring
		-e LD_LIBRARY_PATH=/opt/nccl-mesh
	)
	;;
socket)
	NCCL_ENV+=(-e NCCL_NET=Socket -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="$IFACE")
	;;
*) echo "TRANSPORT must be ib|mesh|socket" >&2; exit 2 ;;
esac

VLLM=/usr/local/lib/python3.12/dist-packages/vllm

# The v12 image patches EAGLE3 aux-hidden-state capture INTO its own copy of
# glm5next/nvidia/model.py, but this mount shadows that file with the TP=3 overlay copy --
# which reverts it, and the engine dies at load with "Model does not support EAGLE3
# interface". Mount a TP=3 copy carrying the same patch instead. Build it with:
#   cp $UPSTREAM_DIR/overlay/vllm/models/glm5next/nvidia/model.py $OVERLAY_DIR/glm5next_model_dflash2.py
#   python3 <overlay-dflash2>/patch_glm_aux_capture.py --model-file $OVERLAY_DIR/glm5next_model_dflash2.py
GLM_MODEL_PY="$UPSTREAM_DIR/overlay/vllm/models/glm5next/nvidia/model.py"
if [ "$SPEC_METHOD" = "dflash" ]; then
	GLM_MODEL_PY="${DFLASH_MODEL_PY:-$OVERLAY_DIR/glm5next_model_dflash2.py}"
	test -f "$GLM_MODEL_PY" || { echo "SPEC_METHOD=dflash needs the aux-capture model.py at $GLM_MODEL_PY (see comment in serve.sh)" >&2; exit 2; }
fi

MOUNTS=(
	-v "$MODEL_DIR:$MODEL_MNT:ro"
	-v "$CACHE_DIR:/cache"
	-v "$GLM_MODEL_PY:$VLLM/models/glm5next/nvidia/model.py:ro"
	-v "$UPSTREAM_DIR/overlay/vllm/model_executor/layers/vocab_parallel_embedding.py:$VLLM/model_executor/layers/vocab_parallel_embedding.py:ro"
	-v "$UPSTREAM_DIR/overlay/vllm/model_executor/model_loader/weight_utils.py:$VLLM/model_executor/model_loader/weight_utils.py:ro"
	-v "$UPSTREAM_DIR/overlay/vllm/model_executor/parameter.py:$VLLM/model_executor/parameter.py:ro"
)
[ "$SPEC_METHOD" = "dflash" ] && MOUNTS+=(-v "$DFLASH2_DIR:$DFLASH2_MNT:ro")
# The GB10 sparse-indexer fallback: without it any request >~25K tokens kills the engine.
# FlashInfer JIT-builds the cutlass fused-MoE kernels at warmup and includes <nvrtc.h>,
# which ships only in the pip nvidia cu13 wheel. Mount that ONE header into nvcc's include
# dir -- putting the whole wheel dir on CPATH shadows the CUDA runtime headers and breaks
# nvcc's generated stubs with "__cudaLaunch was not declared in this scope". DFlash mounts
# it unconditionally; the dflash branch above has already verified it is present and real.
[ "$SPEC_METHOD" = "dflash" ] && MOUNTS+=(-v "$OVERLAY_DIR/nvrtc.h:/usr/local/cuda/include/nvrtc.h:ro")
INDEXER_REL="vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
if [ -f "$OVERLAY_DIR/$INDEXER_REL" ]; then
	MOUNTS+=(-v "$OVERLAY_DIR/$INDEXER_REL:$VLLM/model_executor/layers/sparse_attn_indexer_kpool.py:ro")
else
	echo "WARNING: no indexer overlay at $OVERLAY_DIR/$INDEXER_REL - contexts >~25K will kill the engine" >&2
fi

# Ad-hoc overlays for debugging, e.g.
#   EXTRA_MOUNTS="-v /tmp/kv_cache_utils_debug.py:$VLLM/v1/core/kv_cache_utils.py:ro"
[ -n "${EXTRA_MOUNTS:-}" ] && read -r -a xmnt <<<"$EXTRA_MOUNTS" && MOUNTS+=("${xmnt[@]}")
[ -n "${EXTRA_ENV:-}" ] && read -r -a xenv <<<"$EXTRA_ENV" && NCCL_ENV+=("${xenv[@]}")

ARGS=(
	--gpus all -d --name "$NAME" --restart no
	--network host --ipc host --shm-size 32g
	--ulimit memlock=-1:-1 --cap-add IPC_LOCK
	--device /dev/infiniband:/dev/infiniband
	"${MOUNTS[@]}" "${PLUGIN[@]}"
	-e VLLM_HOST_IP="$HOST_IP"
	-e HF_HOME=/cache/huggingface
	-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
	-e VLLM_ENGINE_READY_TIMEOUT_S=3600
	-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	-e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a
	-e FLASHINFER_DISABLE_VERSION_CHECK=1
	"${NCCL_ENV[@]}"
	"$IMAGE"
	"$MODEL_MNT"
	--served-model-name "$SERVED_NAME"
	--host 0.0.0.0 --port "$PORT"
	--trust-remote-code
	--tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size 1
	--gpu-memory-utilization "$GPU_MEM_UTIL"
	--max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
	--max-num-batched-tokens "$MAX_BATCHED_TOKENS"
	--block-size "$BLOCK_SIZE" --moe-backend "$MOE_BACKEND"
	--kv-cache-dtype "$KV_CACHE_DTYPE"
	"${OPT[@]}"
	--tool-call-parser glm47 --enable-auto-tool-choice
	--reasoning-parser "$REASONING_PARSER"
	--distributed-executor-backend mp
	--nnodes "${#NODE_LIST[@]}" --node-rank "$NODE_RANK"
	--master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
)
[ -n "$HEADLESS" ] && ARGS+=("$HEADLESS")
[ -n "${EXTRA_ARGS:-}" ] && read -r -a extra <<<"$EXTRA_ARGS" && ARGS+=("${extra[@]}")

if [ "$ACTION" = "args" ]; then
	printf '%q ' docker run "${ARGS[@]}"; echo
	exit 0
fi

# GB10's allocator competes with page cache for the KV slab.
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

echo "rank=$NODE_RANK/$((${#NODE_LIST[@]} - 1)) host=$HOST_IP tp=$TP_SIZE transport=$TRANSPORT mtp=$MTP_TOKENS len=$MAX_MODEL_LEN"
docker rm -f "$NAME" 2>/dev/null || true
docker run "${ARGS[@]}"
sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || {
	echo "$NAME exited; docker logs $NAME" >&2
	exit 1
}
