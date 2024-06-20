import subprocess
import sys
import os

# Define the command to be executed
# !python llama.cpp/convert-hf-to-gguf.py ./models/sfr-embedding-mistral/ --outtype f16 --outfile ./models/sfr-embedding mistral.f16.gguf
command = [
    'python',
    'llama.cpp/convert-hf-to-gguf.py',
    './models/sfr-embedding-mistral/',
    '--outtype', 'f16',
    '--outfile', './models/sfr-embedding-mistral.f16.gguf'
]

result = subprocess.run(command, capture_output=True, text=True)

print("Output:", result.stdout)
print("Error:", result.stderr)
