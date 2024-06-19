from transformers import AutoTokenizer, AutoModel
import torch 

tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")

tokenizer.save_pretrained("models/sfr-embedding-mistral")
model.save_pretrained("models/sfr-embedding-mistral")