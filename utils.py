import torch
import bitsandbytes as bnb

def get_completion(query: str, model, tokenizer) -> str:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    prompt_template = """
    <s>
    [INST]
    Below is an instruction that describes a task. Write a response that appropriately completes the request.
    {query}
    [/INST]
    </s>
    <s>

    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

def generate_prompt(data_point):
    """Gen. input text based on a prompt, task instruction, (context info.), and answer

    :param data_point: dict: Data point
    :return: dict: tokenzed prompt
    """
    prefix_text = 'Below is an instruction that describes a task. Write a response that ' \
               'appropriately completes the request.\n\n'
    # Samples with additional context into.
    if data_point['input']:
        text = f"""<s>[INST]{prefix_text} {data_point["instruction"]} here are the inputs {data_point["input"]} [/INST]{data_point["output"]}</s>"""
    # Without
    else:
        text = f"""<s>[INST]{prefix_text} {data_point["instruction"]} [/INST]{data_point["output"]} </s>"""
    return text

def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

