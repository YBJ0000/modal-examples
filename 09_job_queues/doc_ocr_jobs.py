# ---
# deploy: true
# mypy: ignore-errors
# ---

# # Run a job queue for olmOCR

# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](https://modal.com/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).

# Our job queue will handle a single task: running OCR transcription for images of receipts and documents.
# We'll make use of olmOCR: a toolkit for converting PDFs and other image-based document formats 
# into clean, readable, plain text format using a 7B parameter Vision Language Model.

# ## Define an App

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).
# Later, we'll use the name provided for our `App` to find it from our web app and submit tasks to it.

from typing import Optional
import modal

app = modal.App("example-doc-ocr-jobs")  # 改为与 GOT-OCR 版本相同的名称

# We define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).
# olmOCR requires specific dependencies including poppler-utils for PDF processing.

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "poppler-utils",
        "curl",
        "fonts-liberation",
        "fonts-dejavu-core",
        "fontconfig"
    )
    # Install PyTorch with CUDA support first
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        index_url="https://download.pytorch.org/whl/cu121"
    )
    # Install other dependencies from PyPI
    .pip_install(
        "accelerate==0.28.0",
        "huggingface_hub[hf_transfer]==0.27.1",
        "transformers==4.48.0",
        "numpy<2",
        "safetensors",
        "Pillow"
    )
    # Install latest olmOCR with GPU support and flash infer for faster inference
    .pip_install("olmocr[gpu]>=0.2.1")
    # Install flash infer for faster GPU inference (recommended by olmOCR)
    .pip_install("https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl")
)

# ## Cache the pre-trained model on a Modal Volume

# olmOCR uses models that are downloaded and cached automatically.
# We create a Modal [Volume](https://modal.com/docs/guide/volumes) to store the model cache
# to avoid re-downloading on every request.

model_cache = modal.Volume.from_name("olmocr-cache", create_if_missing=True)

# Use a dedicated path for the model cache that won't conflict with existing directories
MODEL_CACHE_PATH = "/root/model_cache"
inference_image = inference_image.env({
    "HF_HOME": MODEL_CACHE_PATH,
    "TRANSFORMERS_CACHE": MODEL_CACHE_PATH,
    "HF_HUB_CACHE": MODEL_CACHE_PATH,
    "HF_HUB_ENABLE_HF_TRANSFER": "1"
})

# ## Run OCR inference on Modal using optimized approach

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator with model caching for optimal performance.

# olmOCR requires a GPU with at least 20GB of GPU RAM. We use A100 which has 40GB.

@app.function(
    gpu="a100",
    retries=3,
    volumes={MODEL_CACHE_PATH: model_cache},
    image=inference_image,
    timeout=600,  # 10 minutes timeout for processing
)
def parse_receipt(image: bytes) -> str:
    """
    Optimized olmOCR implementation using Python API with model caching.
    This avoids the overhead of subprocess calls and model reloading.
    """
    import tempfile
    import base64
    import torch
    from io import BytesIO
    from PIL import Image
    from pathlib import Path
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    
    # Import olmOCR modules
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_finetuning_prompt
    from olmocr.prompts.anchor import get_anchor_text
    
    # Load model and processor (will be cached by Modal)
    print("Loading olmOCR model and processor...")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.bfloat16
    ).eval()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Model loaded successfully on {device}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Determine file type and save appropriately
            if image.startswith(b'%PDF'):
                # PDF file
                pdf_path = temp_path / "input.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(image)
                
                # Render PDF page to base64 image (fixed parameter order)
                image_base64 = render_pdf_to_base64png(
                    str(pdf_path), 
                    1,  # page number as positional argument
                    target_longest_image_dim=1024
                )
                
                # Get anchor text for document metadata (fixed parameter order)
                anchor_text = get_anchor_text(
                    str(pdf_path), 
                    1,  # page number as positional argument
                    pdf_engine="pdfreport", 
                    target_length=4000
                )
                
            elif image.startswith(b'\x89PNG') or image.startswith(b'\xff\xd8\xff'):
                # PNG or JPEG image
                pil_image = Image.open(BytesIO(image))
                
                # Convert to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # For images, use minimal anchor text
                anchor_text = ""
                
            else:
                return "Error: Unsupported file format. Please provide PDF, PNG, or JPEG files."
            
            # Build the prompt using olmOCR's prompt building system
            prompt = build_finetuning_prompt(anchor_text)
            
            # Build the full prompt for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]
            
            # Apply chat template and process
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Decode base64 image for processing
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
            
            # Process inputs
            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}
            
            print("Running olmOCR inference...")
            
            # Generate output using the model
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    temperature=0.8,
                    max_new_tokens=2048,  # Increased for longer documents
                    num_return_sequences=1,
                    do_sample=True,
                )
            
            # Decode the output
            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            text_output = processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            
            result = text_output[0] if text_output else "No output generated"
            print(f"olmOCR processing completed. Output length: {len(result)} characters")
            
            return result
            
    except Exception as e:
        error_msg = f"olmOCR processing failed: {str(e)}"
        print(error_msg)
        return error_msg

# ## Deploy

# Now that we have a function, we can publish it by deploying the app:

# ```shell
# modal deploy doc_ocr_jobs.py
# ```

# Once it's published, we can [look up](https://modal.com/docs/guide/trigger-deployed-functions) this Function
# from another Python process and submit tasks to it:

# ```python
# fn = modal.Function.from_name("example-doc-ocr-jobs", "parse_receipt")
# fn.spawn(my_image)
# ```

# Modal will auto-scale to handle all the tasks queued, and
# then scale back down to 0 when there's no work left.

# ## Run manually

# We can also trigger `parse_receipt` manually for easier debugging:

# ```shell
# modal run doc_ocr_jobs.py
# ```

@app.local_entrypoint()
def main(receipt_filename: Optional[str] = None):
    import urllib.request
    from pathlib import Path

    if receipt_filename is None:
        receipt_filename = Path(__file__).parent / "receipt.png"
    else:
        receipt_filename = Path(receipt_filename)

    if receipt_filename.exists():
        image = receipt_filename.read_bytes()
        print(f"running optimized olmOCR on {receipt_filename}")
    else:
        # Use a sample PDF for testing
        receipt_url = "https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running optimized olmOCR on sample PDF from URL {receipt_url}")
    
    result = parse_receipt.remote(image)
    print("=" * 50)
    print("OCR Result:")
    print("=" * 50)
    print(result)
