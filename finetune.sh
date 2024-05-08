#!/bin/bash

accelerate launch --mixed_precision=fp16 --num_processes=1 finetuning.py \
    --model_name_or_path="mistralai/Mistral-7B-v0.1" \
    --report_to="none" \
    --dataset_name="TokenBender/code_instructions_122k_alpaca_style" \
    --dataset_text_field="prompt" \
    --max_seq_length=2048 \
    --num_train_epochs=3 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16 \
    --load_in_4bit=True \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --warmup_ratio=0.03 \
    --max_steps=10 \
    --learning_rate=2e-4 \
    --logging_steps=1 \
    --output_dir="mistral_outputs" \
    --optim="paged_adamw_8bit" \
    --save_strategy="epoch" \
    --fp16=True \