"""Pre-download the BLIP-2 retrieval weights into the image's HuggingFace cache at
build time, so the running tagger never downloads at request time.
Invoked by the Containerfile."""

# read the model id from config.yml so the baked weights always match run.py
import yaml
from transformers import Blip2Processor, Blip2ForImageTextRetrieval

with open("config.yml") as f:
    model_id = yaml.safe_load(f)["model"]["model_id"]

print(f"prefetching {model_id} into the HF cache ...", flush=True)
Blip2Processor.from_pretrained(model_id)
Blip2ForImageTextRetrieval.from_pretrained(model_id)
print("done", flush=True)
