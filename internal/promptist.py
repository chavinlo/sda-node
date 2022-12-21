from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompter():
  prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text, prompter_model, prompter_tokenizer):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    # Just use 1 beam and get 1 output, this is much, much, much faster than 8 beams and 8 outputs and we're only using the first.
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    # Use [input_ids.shape[-1]:] because the decoded tokenised version of plain_text may have a different number of characters to the original
    res = prompter_tokenizer.decode(outputs[0][input_ids.shape[-1]:])
    return res