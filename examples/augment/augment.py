from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

model_name = "tuner007/pegasus_paraphrase"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to("cpu")

text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to("cpu")

# paraphrased = model.generate(**inputs, num_return_sequences=1, num_beams=5)
# result = tokenizer.decode(paraphrased[0], skip_special_tokens=True)
#
# print(f"Original: {text}")
# print(f"Paraphrased: {result}")

for i in range(5):
    paraphrased = model.generate(
        **inputs, 
        num_return_sequences=1, 
        do_sample=True, 
        num_beams=10, 
        temperature=1.5, 
        max_length=60
    )
    result = tokenizer.decode(paraphrased[0], skip_special_tokens=True)
    print(f"Paraphrased: {result}")
