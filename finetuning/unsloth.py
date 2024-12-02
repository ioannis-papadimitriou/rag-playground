# Install required packages
"""
!pip install @ git+https://github.com/unslothai/unsloth.git@a2f4c9793ecf829ede2cb64f2ca7a909ce3b0884
!pip install torch
!pip install transformers
!pip install datasets
!pip install trl
!pip install accelerate -U
!pip install bitsandbytes
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
import json
from trl import SFTTrainer
from transformers import TrainingArguments
import gc

# Configuration
max_seq_length = 2048
load_in_4bit = True
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

# Training parameters
rank = 32
lora_alpha = 4
use_rslora = False
learning_rate = 4e-5
dataset_num_proc = 4
per_device_batch_size = 1
gradient_accumulation_steps = 2
save_steps = 2000
num_train_epochs = 1
packing = True
output_dir = './outputs'

# Load and prepare the model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj",
                     "embed_tokens", "lm_head"],
    lora_alpha = lora_alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = use_rslora,
)

# Load and prepare the dataset
def load_qa_pairs(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract QA pairs from the data structure
    qa_pairs = []
    for item in data['qa_pairs']:
        qa_pairs.append({
            'question': item['question'],
            'answer': item['answer'],
            'context': item['context']
        })
    
    return Dataset.from_list(qa_pairs)

# Load your data
dataset = load_qa_pairs('your_qa_pairs_file.json')

# Prepare chat template
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful AI assistant that provides accurate responses based on the given context.
{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context}
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{answer}<|eot_id|>"""

# Apply chat template
from unsloth import apply_chat_template

def format_dialog(sample):
    return {
        'text': chat_template.format(
            SYSTEM="Answer questions accurately based on the provided context.",
            context=sample['context'],
            question=sample['question'],
            answer=sample['answer']
        )
    }

dataset = dataset.map(format_dialog)

# Initialize trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = dataset_num_proc,
    packing = packing,
    args = TrainingArguments(
        save_strategy="steps",
        save_steps = save_steps,
        per_device_train_batch_size = per_device_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 5,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not FastLanguageModel.is_bfloat16_supported(),
        bf16 = FastLanguageModel.is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
    ),
)

# Train the model
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("lora_model_qa_finetuned")
tokenizer.save_pretrained("lora_model_qa_finetuned")

# Optional: Function for inference
def run_inference(question, context):
    messages = [
        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors = "pt"
    ).to("cuda")
    
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    FastLanguageModel.for_inference(model)
    output = model.generate(
        input_ids, 
        streamer=text_streamer,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)