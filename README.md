# LLM Fine-Tuning Project

Welcome to the LLM Fine-Tuning project! This repository contains scripts to fine-tune different Language Model Models (LLMs) on various datasets.

## Getting Started

Follow these instructions to clone the repository, set up the Conda environment, and run the scripts:

### Prerequisites

- Git
- Conda

### Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/debjyotiSRoy/llm_finetuning.git
```

### Set Up Conda Environment

Update cuda-toolkit: [Optional]

```bash
sudo apt install nvidia-cuda-toolkit
```
Navigate to the cloned repository directory:

```bash
cd llm_finetuning
git checkout master
```

Create a Conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

Activate the Conda environment:

```bash
conda activate mistralai
```

Install dependencies:

```bash
pip install -r torch_requirements.txt
pip install -r requirements.txt
```
Accelerate Config:
```bash
accelerate config # will prompt you to define the training configuration
```

### Login Hugging Face

```bash
huggingface-cli login
```
Then get your WRITE token from [hugging face token](https://huggingface.co/settings/tokens) and paste it.

### Run the Scripts

#### Training
```bash
export TRL_USE_RICH=1
```
In the file `finetune.sh` replace `"model_name_or_path"`, `"dataset_name"`, and `"output_dir"` with the appropriate values. Note that this script is tested on GPU with at least 32GB of VRAM. You may need to adjust the parameter `"max_seq_length"` if you are training on a GPU(s) with lesser memory. If you want to perform multi-GPU training then just adjust the `"num_processes"` parameter accordingly. 

At the end of the training, it asks you for a prompt, wherte you can type something like "generate python code for fibonacci series using recusrion". Try this out. Note that, to be to be able to generate really good responses you might need to train for `"num_train_epochs=5"` or `max_steps=500`. Though you might ocassionally see some good responses for lesser `"num_train_epochs"` or `"max_steps"`.

```bash
./finetune.sh
```
#### Inference
TBD

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## Acknowledgements

We would like to thank Alberto Ceballos Arroyo for providing the Conda environment setup. We also acknowledge the notebook at [LLM-Alchemy-Chamber](https://github.com/adithya-s-k/LLM-Alchemy-Chamber/blob/main/Finetuning/Mistral_finetuning_notebook.ipynb) by Adithya S Kolavi, which we have used to get the code for training the LLM Mistral.

## License

This project is licensed under the [Apache License](LICENSE).
