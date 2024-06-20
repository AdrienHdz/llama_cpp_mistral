from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

model_original = SentenceTransformer("models/sfr-embedding-mistral")
model = Llama("models/sfr-embedding-mistral.f16.gguf", embedding=True)
modeltrue = Llama("ggml-sfr-embedding-mistral-f16.gguf", embedding=True)

embed_original = model_original.encode(["hello world"])
embed = model.embed("hello world")
embedtrue = modeltrue.embed("hello world")

print(embed_original.shape)
print(embed.shape)
print(embedtrue.shape)

