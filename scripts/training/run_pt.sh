# 运行脚本前请仔细阅读wiki(https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh)
lr=2e-4
lora_rank=64
lora_alpha=128
lora_dropout=0.05
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"                # 嵌入层-输入 + 连接层-输出, 可删去脚本中的 modules_to_save, 不训练 embed_tokens,lm_head, 只训练 LoRA 参数

pretrained_model=./base/chinese-alpaca-2-7b           # meta-llama2-hf/chinese-llama2/chinese-alpaca2
chinese_tokenizer_path=./base/chinese-alpaca-2-7b     # chinese-llama2 的 tokenizer(55296)
dataset_dir=./data/pt/xxx                             # 预训练数据的目录, 可包含多个 .txt 纯文本文件
data_cache=./data/pt/cache/xxx                        # 存放数据缓存文件的目录
output_dir=./experiments/outputs/pt                   # 训练后的 LoRA 权重和配置存放于 ${output_dir}/pt_lora_model, 可用于后续的合并流程
per_device_train_batch_size=1
gradient_accumulation_steps=8
block_size=512                                        # 类同 max_length

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nnodes 1 --nproc_per_node 1 run_clm_pt_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${chinese_tokenizer_path} \
    --dataset_dir ${dataset_dir} \
    --data_cache_dir ${data_cache} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --do_train \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 200 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 8 \
    --block_size ${block_size} \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --lora_dropout ${lora_dropout} \
    --modules_to_save ${modules_to_save} \
    --torch_dtype float16 \
    --load_in_kbits 16 \
    --save_safetensors False \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False
