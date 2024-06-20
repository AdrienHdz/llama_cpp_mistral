from llama_cpp import Llama
model = Llama("models/sfr-embedding-mistral.f16.gguf", embedding=True)
modeltrue = Llama("ggml-sfr-embedding-mistral-f16.gguf", embedding=True)

embed = model.embed("hello world")
embedtrue = model.embed("hello world")

print(embed == embedtrue)