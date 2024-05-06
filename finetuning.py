"""
# peft:
python finetune_mistral.py \
    --model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import init_zero_verbose, SftScriptArguments, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import Dataset, load_dataset

from tqdm.rich import tqdm
from pathlib import Path
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

from trl import (
    ModelConfig,
    RichProgressCallback,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

from utils import get_completion, generate_prompt

tqdm.pandas()

if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((SftScriptArguments, TrainingArguments, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
        # Model Loading
    ################
    # We'll load the model using QLoRA quantization to reduce the usage of memory
    quantization_config = get_quantization_config(model_config)
    quantization_config.update(bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)

    # Now we specify the model ID and then we load it with our previously defined quantization configuration.
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, quantization_config=quantization_config, device_map=get_kbit_device_map())
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Run a inference on the base model. The model does not seem to understand our instruction and gives us a list of questions related to our query.
    # result = get_completion(query="code the fibonacci series in python using reccursion", model=model, tokenizer=tokenizer)
    # print(result)

    ################
    # Dataset
    ################
    # raw_dataset = load_dataset(args.dataset_name, split="train")
    # <remove following later: just to load locally from cache fro speed>
    raw_dataset = Dataset.from_file("/home/deb/.cache/huggingface/datasets/TokenBender___code_instructions_122k_alpaca_style/default/0.0.0/19b59da67914b5fb2e0a5dff937e9917c0cfb7e4/code_instructions_122k_alpaca_style-train.arrow")
    """
    We'll put each instruction and input pair between [INST] and [/INST] output after that, like this:
    <s>[INST] What is your favorite condiment? [/INST]
    Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavor to whatever I'm cooking up in the kitchen!</s>
    """
    text_column = [generate_prompt(data_point) for data_point in raw_dataset]
    # add the "prompt" column in the dataset
    dataset = raw_dataset.add_column("prompt", text_column)
    # We'll need to tokenize our data so the model can understand.
    dataset = dataset.shuffle(seed=1234)  # Shuffle dataset here
    dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)
    # Split dataset into 90% for training and 10% for testing
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset[args.dataset_train_name]
    test_dataset = dataset[args.dataset_test_name]

    ################
    # Apply Lora
    ################
    """
    Here comes the magic with peft! Let's load a PeftModel and specify that we are going to use low-rank adapters (LoRA) using get_peft_model 
    utility function and  the prepare_model_for_kbit_training method from PEFT.
    """
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    """
    Use the following function to find out the linear layers for fine tuning. 
    QLoRA paper : "We find that the most critical LoRA hyperparameter is how many LoRA adapters are 
    used in total and that LoRA on all linear transformer block layers is required to match full finetuning performance."
    """
    from utils import find_all_linear_names
    modules = find_all_linear_names(model)
    print(f"The modules of the pretrained LLM on which we will apply LoRA: {modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the SFTTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Training
    ################
    with init_context:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field=args.dataset_text_field,
            max_seq_length=args.max_seq_length,
            peft_config=lora_config,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()

    ################
    # Saving
    ################
    project = "mistral-codeset-finetuned"
    # run_name = model_config.model_name_or_path + "-" + project
    from datetime import datetime
    # finetuned_model_name = Path(training_args.output_dir)/f"{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}" #Name of the model you will be pushing to huggingface model hub
    finetuned_model_name = Path(training_args.output_dir)/project  # Name of the model you will be pushing to huggingface model hub
    with save_context:
        trainer.model.save_pretrained(finetuned_model_name)

    ################
    # Merging and Sharing on Hub
    ################
    base_model =  AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    merged_model= PeftModel.from_pretrained(base_model, finetuned_model_name)
    merged_model= merged_model.merge_and_unload()

    # Save the merged model
    finetuned_merged_model = Path(training_args.output_dir)/'-'.join(finetuned_model_name.name.split('-')[:3] + ['merged'] + finetuned_model_name.name.split('-')[3:])
    merged_model.save_pretrained(finetuned_merged_model,safe_serialization=True)
    tokenizer.save_pretrained(finetuned_merged_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Push the model and tokenizer to the HuggingFace Model Hub [Optional]
    # merged_model.push_to_hub("deb101/mistral_outputs", use_temp_dir=False)
    # tokenizer.push_to_hub("deb101/mistral_outputs", use_temp_dir=False)


