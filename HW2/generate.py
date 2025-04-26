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
OUTPUT_DIR = "pdf_pages"        # Where to save images and JSON
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

SYSTEM_PROMPT = """
You are an AI lecture slide analyzer. The following input is an image and the OCR of a lecture slide about “Artificial Intelligence.”

1. Extract every piece of written content:
   • Slide title
   • Sub-bullets and their full text
   • Definitions, formulas, and any inline examples

2. The summary should be as thorough and precise as possible—this will be used for later retrieval and generation.

3. The keywords should be the most relevant terms from the slide, containing at least 5 terms.

4. If the slide shows a plot, describe the plot.

5. Organize your output as a valid JSON object with these fields:
   {
     "title": string,
     "summary": string,
     "definitions": { term: definition },
     "keywords": [ string ],
     "formulas": [ string ]
   }

6. Output ONLY the JSON object as a string—no extra text or formatting.

7. The formulas should be in LaTeX format, and the JSON object should be valid and parsable, don't generate \u00b7.

"""

# --- Model Setup ---
generation_config = GenerationConfig.from_pretrained(MODEL_ID)
generation_config.max_new_tokens = 1024
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype="auto",
    _attn_implementation="flash_attention_2"
).to("cuda")

def caption_with_phi4(img: Image.Image, system: str) -> str:
    prompt = (
        "<|im_start|>system<|im_sep|>"
        + system.strip()
        + "<|image_1|><|im_end|>"
        + "<|im_start|>assistant<|im_sep|>"
    )
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=generation_config.max_new_tokens,
        )
    return processor.decode(outputs[0], skip_special_tokens=True)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

failed = []
# --- Main Loop: process each page ---
reader = PdfReader(FILE)
for page_num in range(1, len(reader.pages) + 1):
    page_num = 393
    # 1. Render PDF page to image
    images = convert_from_path(
        FILE, dpi=200,
        first_page=page_num, last_page=page_num,
        use_pdftocairo=True
    )
    img = images[0]
    base = f"page_{page_num:03d}"

    # 2. Generate raw caption
    raw_caption = caption_with_phi4(img, SYSTEM_PROMPT)
    # save raw caption
    with open(os.path.join(OUTPUT_DIR, base + "_caption.json"), "w", encoding="utf-8") as f:
        f.write(raw_caption)

    # 3. Extract JSON payload
    #   a) Try to capture fenced ```json ... ``` block
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw_caption)
    if m:
        json_str = m.group(1)
    else:
        # b) Fallback: grab from first '{' to last '}'
        start = raw_caption.find('{')
        end   = raw_caption.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = raw_caption[start:end]
        else:
            print(f"❌ page_{page_num:03d}: could not locate JSON block")
            continue

    # 4. Parse and save
    try:
        data = json.loads(json_str)

        # save stripped JSON
        with open(os.path.join(OUTPUT_DIR, base + "_caption_strip.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # save page image
        img.save(os.path.join(OUTPUT_DIR, base + ".png"))
        print(f"✅  page_{page_num:03d} processed")

    except json.JSONDecodeError as e:
        print(f"❌ page_{page_num:03d} JSON decode error: {e}")
        failed.append(page_num)
        # optionally bump generation_config.max_new_tokens and retry here
    finally:
        # cleanup
        del img, images
        gc.collect()

# save failed into txt
with open(os.path.join(OUTPUT_DIR, "failed.txt"), "w") as f:
    for page_num in failed:
        f.write(f"{page_num}\n")