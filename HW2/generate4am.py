import os
import gc
import json
import re
from pdf2image import convert_from_path
from PIL import Image
# import pytesseract  # uncomment if you need OCR
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PyPDF2 import PdfReader
import torch

# --- Configuration ---
FILE = "AI.pdf"                 # Path to your PDF
IMG_DIR = "pdf_pages"        # Where to save images
OUTPUT_DIR = "pdf_pages3"        # Where to save images and JSON
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

SYSTEM_PROMPT = """
You are a professor of Artificial Intelligence. The following input is an image of a lecture slide.

When explaining the slide, speak in a clear, pedagogical style appropriate for graduate students. Your explanation should include:
1. Every piece of written content: slide title, sub-bullets with full text, definitions, formulas, and any inline examples.  
2. A detailed narrative of at least 300 words that walks students through the key ideas on the slide.  
3. At least 5 of the most relevant keywords from the slide, highlighted and briefly defined.  
4. If the slide shows a plot, a concise description of what the plot illustrates and how it supports the slide's message.  
5. If the slide contains any formulas, present them in LaTeX format and then unpack each component, explaining its meaning and role in context.
6. It should be able to answer questions about the slide, such as "What is the main idea of this slide?" or "What is the significance of the formula on this slide?".
"""

# --- Model Setup ---
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
generation_config.max_new_tokens = 1024
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
).to("cuda")

def caption_with_phi4(img: Image.Image, system: str) -> str:
    prompt = (
        "<|im_start|>system<|im_sep|>"
        + system.strip()
        + "<|im_start|>user<|im_sep|>I'm a student learning artificial intelligence, teach me every thing in this slide.<|im_end|>"
        + "<|image_1|><|im_end|>"
        + "<|im_start|>assistant<|im_sep|>"
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.2,     # 保持一致性
            num_beams=4,          # 提高质量
            no_repeat_ngram_size=3
        )
    return processor.decode(outputs[0], skip_special_tokens=True)

def extract_json(raw_caption: str) -> str:
    # 1) Try regex for the assistant block
    m = re.search(r"<\|im_start\|>assistant<\|im_sep\|>(\{[\s\S]*\})", raw_caption)
    if m:
        return m.group(1).strip()
    # 2) Fallback: split on the last separator
    parts = raw_caption.rsplit("<|im_start|>assistant<|im_sep|>", 1)
    return parts[-1].strip()

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

failed = []
# --- Main Loop: process each page ---
reader = PdfReader(FILE)

for page_num in range(1, len(reader.pages) + 1):
    if (page_num <= 174) and (page_num >= 103):
        continue
    # read img 
    img = Image.open(IMG_DIR + f"/page_{page_num:03d}.png")
    base = f"page_{page_num:03d}"

    # 2. Generate raw caption
    raw_caption = caption_with_phi4(img, SYSTEM_PROMPT)
    # save raw caption
    with open(os.path.join(OUTPUT_DIR, base + "_caption.txt"), "w", encoding="utf-8") as f:
        f.write(raw_caption)

    outputstr = extract_json(raw_caption)
    print(f"outputstr: {outputstr}")
    # 4. Parse (to verify) and save raw JSON text into .txt
    try:
        # save the raw JSON string into a .txt file
        txt_path = os.path.join(OUTPUT_DIR, base + "_caption_strip.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(outputstr)

        print(f"✅  page_{page_num:03d} processed, saved to {txt_path}")

    except json.JSONDecodeError as e:
        print(f"❌ page_{page_num:03d} decode error: {e}")
        failed.append(page_num)
    finally:
        # cleanup
        del img
        gc.collect()


# save failed into txt
with open(os.path.join(OUTPUT_DIR, "failed.txt"), "w") as f:
    for page_num in failed:
        f.write(f"{page_num}\n")