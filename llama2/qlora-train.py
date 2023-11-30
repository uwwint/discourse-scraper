import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import get_peft_config, prepare_model_for_kbit_training, get_peft_model, LoraConfig
from trl import SFTTrainer
import time

# Load conversation dataset
conv_dataframe = pd.read_csv("../data/all-conv-galaxy-q-a.csv", sep="\t")
print("Size of data: {}".format(len(conv_dataframe)))

# Split dataset into training and evaluation sets
tr_index = 1800
final_index = len(conv_dataframe)
tr_conv = conv_dataframe[:tr_index]
eval_conv = conv_dataframe[tr_index + 1: final_index]
print("Size of tr/te: {}/{}".format(len(tr_conv), len(eval_conv)))
dataset = Dataset.from_pandas(tr_conv).train_test_split(test_size=0.2, seed=42)

# Save evaluation dataset to a CSV file
eval_conv.to_csv("../data/eval_dataset.csv", sep="\t", index=None)

# Load pre-trained model and tokenizer
model_name = "NousResearch/Llama-2-7b-chat-hf"
compute_dtype = torch.float16
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, use_cache=False, device_map="auto")
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
#target_modules = ['q_proj','v_proj', 'k_proj', 'o_proj']
target_modules = ["q_proj","v_proj"]

# Load LoRA configuration
peft_config = LoraConfig(lora_alpha=32, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM",
                          target_modules=target_modules)

print("Extracting parameter efficient model ...")
start_time = time.time()
refined_model = get_peft_model(prepare_model_for_kbit_training(model), peft_config)
end_time = time.time()
refined_model.print_trainable_parameters()
print(f"PEFT loading time: {end_time - start_time} seconds")

base_dir = "llama-linear-layers-all-conv-Nov-30-2"

print("Setting up Training arguments ...")

# Set up training arguments
training_arguments = TrainingArguments(
    output_dir=base_dir,
    evaluation_strategy="steps",
    do_eval=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    optim="adamw_hf",
    save_steps=50,
    logging_steps=50,
    eval_steps=50,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=4,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

print("Setting up SFTTrainer ...")

start_time = time.time()

# Set up SFTTrainer
trainer = SFTTrainer(
    model=refined_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    dataset_text_field="conversations",
    max_seq_length=700,
    tokenizer=tokenizer,
    args=training_arguments,
)

end_time = time.time()
print(f"SFTTTrainer setting up time: {end_time - start_time} seconds")

print("Start training ...")
trainer.train()
trainer.save_model()

# Save the refined model
#refined_model.config.to_json_file("saved-model/config.json")
#refined_model.save_pretrained('saved-model')

print("Finished training ...")
