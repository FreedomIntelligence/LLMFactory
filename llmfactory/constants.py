IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

TRAIN_SCRIPT = r"""
model_name_or_path=<<model_path>>
model_max_length=2048
data_path=<<data_path>>
output_dir=<<output_dir>>
fsdp_transformer_layer_cls_to_wrap=<<fsdp_transformer_layer_cls_to_wrap>>

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=12375 \
  train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 32 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap ${fsdp_transformer_layer_cls_to_wrap} \
  --tf32 True \
  --gradient_checkpointing True
"""