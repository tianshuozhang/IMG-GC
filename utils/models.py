from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
# from dataset import ImageCaptioningDataset
from tqdm import tqdm

def get_processor_and_model(processorpath="Salesforce/blip2-opt-2.7b",modelpath="Salesforce/blip2-opt-2.7b",device_map="auto"):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    processor = Blip2Processor.from_pretrained(processorpath)
    model = Blip2ForConditionalGeneration.from_pretrained(modelpath,device_map=device_map,torch_dtype=torch.float16)
    return processor,model
def lora_fit(model):
    config = LoraConfig(        #lora 训练时，所有的参数均被冻结
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

def train_with_lora(model,dataset,processor,epochs=15,device="cuda"):
    def collator(batch):
    # pad the input_ids and attention_mask
        processed_batch = {}
        for key in batch[0].keys():
            if key != "text":
                processed_batch[key] = torch.stack([example[key] for example in batch])
            else:
                text_inputs = processor.tokenizer(
                    [example["text"] for example in batch], padding=True, return_tensors="pt"
                )
                processed_batch["input_ids"] = text_inputs["input_ids"]
                processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch
    train_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    loss_list = []
    for epoch in range(epochs):
        print("Epoch:", epoch)
        sum_loss_list = []

        for idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)
            #语言模型输入=输出
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)

            loss = outputs.loss

            # print("Loss:", loss.item())
            tqdm.set_description(f"Loss: {loss:.4f}")
            sum_loss_list.append(float(loss.item()))

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if idx % 10 == 0:
                generated_output = model.generate(pixel_values=pixel_values)
                print(processor.batch_decode(generated_output, skip_special_tokens=True))

        avg_sum_loss = sum(sum_loss_list) / len(sum_loss_list)
        print("epoch: ", epoch, "loss: ", float(avg_sum_loss))
        loss_list.append(float(avg_sum_loss))

    

if __name__=="main":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
