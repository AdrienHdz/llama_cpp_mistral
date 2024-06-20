from transformers import AutoTokenizer, AutoModel
import torch 
import subprocess

tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")

tokenizer.save_pretrained("models/sfr-embedding-mistral")
model.save_pretrained("models/sfr-embedding-mistral")

# Download sfr-embedding-mistral.gguf from hugginface for comparaison
#!wget https://huggingface.co/dranger003/SFR-Embedding-Mistral-GGUF/resolve/main/ggml-sfr-embedding-mistral-f16.gguf?download=true
url = "https://huggingface.co/dranger003/SFR-Embedding-Mistral-GGUF/resolve/main/ggml-sfr-embedding-mistral-f16.gguf?download=true"

output_filename = "ggml-sfr-embedding-mistral-f16.gguf"

command = [
    'wget',
    url,
    '-O', output_filename 

result = subprocess.run(command, capture_output=True, text=True)

print("Output:", result.stdout)
print("Error:", result.stderr)