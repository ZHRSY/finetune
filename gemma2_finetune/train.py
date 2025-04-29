# 加载模型和分词器
from unsloth import FastLanguageModel
from local_dataset import LocalJsonDataset
from safetensors.torch import load_model, save_model

max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./gemma-2-2b-it-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# 加载和预处理数据集
custom_prompt = """你是西游记里面的孙悟空，请回答问题
### 问题:
{}
### 答案:
{}"""

custom_dataset = LocalJsonDataset(json_file='西游记_训练样本.json', data_template=custom_prompt, tokenizer=tokenizer, max_seq_length=max_seq_length)
dataset = custom_dataset.get_dataset()


# 设置训练配置
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=500,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs"
    ),
)



# 训练模型
trainer.train()

print("训练完成")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("模型保存完成")

# 量化
model.save_pretrained_gguf("model", tokenizer,)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")