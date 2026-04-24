#!/usr/bin/bash

params=""
if [ $# -ne 0 ]; then
    params="$*"
fi

# use envs as local params for convenience
# e.g.
# NNODE=1 NGPU=8 LOG_RANK=0 ./train.sh
NNODE=${NNODE:-"1"}
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0}

if [[ -z "${MASTER_ADDR}" ]]; then
  export MASTER_ADDR="localhost"
fi
if [[ -z "${MASTER_PORT}" ]]; then
  export MASTER_PORT="0"
fi

: '
Usage:

bash train.sh -h

Training a 340M model:

NNODE=1 NGPU=8 LOG_RANK=0 bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/transformer-340M-10B/batch32.seqlen2048.warmup1024.update1.steps20480.lr3e-4 \
  --model.config configs/transformer_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 3e-4 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 32 \
  --training.seq_len 2048 \
  --training.gradient_accumulation_steps 1 \
  --training.steps 20480 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name default \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --training.tensor_parallel_degree 1 \
  --training.disable_loss_parallel \
  --checkpoint.interval 2048 \
  --checkpoint.load_step -1 \
  --metrics.log_freq 1
'

echo "Launching training..."

set -x
path=$(grep -oP '(?<=--job.dump_folder )[^ ]+' <<< "$params")
steps=$(grep -oP '(?<=--training.steps )[^ ]+' <<< "$params")
config=$(grep -oP '(?<=--model.config )[^ ]+' <<< "$params")
tokenizer=$(grep -oP '(?<=--model.tokenizer_path )[^ ]+' <<< "$params")
# Extract model_type directly from the JSON config to skip a full Python
# cold-start (~5s) on every launch. Falls back to the Python import path if
# the JSON has a nested HF auto_map or similar that jq can't resolve.
model=$(jq -r '.model_type // empty' "$config" 2>/dev/null)
if [[ -z "$model" ]]; then
  model=$(
    python -c "import sys; sys.path.insert(0, '$(dirname $0)/..'); import fla, powerformer_hf, powerssm; from transformers import AutoConfig; print(AutoConfig.from_pretrained(sys.argv[1]).to_json_string())" "$config" | jq -r '.model_type'
  )
fi

mkdir -p $path
# Archive a code snapshot into the exp dir for reproducibility. Skip the copy
# entirely on re-launch / resume to save several seconds of I/O per startup.
if [[ ! -d "$path/flame" ]]; then
  echo "Archiving code snapshot under $path (first launch)..."
  for f in *; do
    [[ -f "$f" ]] && cp "$f" "$path/"
  done
  cp -r configs "$path"
  cp -r flame   "$path"
  cp -r 3rdparty/flash-linear-attention/fla "$path"
  cp -r 3rdparty/torchtitan/torchtitan "$path"
else
  echo "Code snapshot already present under $path — skipping copy (set -f $path to force refresh)."
fi

# for offline systems
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1
# export HF_HUB_OFFLINE=1
if [ "$date" == "" ]; then
  date=$(date +%Y%m%d%H%M)
fi
RUN_NAME="$model-$(basename $path)"
RUN_ID="$RUN_NAME-$date"

export WANDB_RESUME=allow
if [[ -z "${WANDB_PROJECT}" ]]; then
  export WANDB_PROJECT="fla"
fi
if [[ -z "${WANDB_NAME}" ]]; then
  export WANDB_NAME="$RUN_NAME"
fi
if [[ -z "${WANDB_RUN_ID}" ]]; then
  export WANDB_RUN_ID="$RUN_ID"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
python -m torch.distributed.run --nnodes=${NNODE} \
  --nproc_per_node=${NGPU} \
  --rdzv_backend c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank \
  --tee 3 \
  --log-dir $path/logs \
  -m flame.train \
  $params

echo "TRAINING DONE!"
echo "Converting the DCP checkpoints to HF format..."

python -m flame.utils.convert_dcp_to_hf \
  --path $path \
  --step $steps \
  --config $config \
  --tokenizer $tokenizer

echo "RUNNING DONE!"
