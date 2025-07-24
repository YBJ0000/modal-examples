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
    # Install olmOCR
    .pip_install("olmocr[gpu]")
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

# ## Run OCR inference on Modal by wrapping with `app.function`

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator, we set up a Modal [Function](https://modal.com/docs/reference/modal.Function).
# We provide arguments to that decorator to customize the hardware, scaling, and other features
# of the Function.

# olmOCR requires a GPU with at least 20GB of GPU RAM. We use A100 which has 40GB.


@app.function(
    gpu="a100",
    retries=3,
    volumes={MODEL_CACHE_PATH: model_cache},
    image=inference_image,
    timeout=600,  # 10 minutes timeout for processing
)
def parse_receipt(image: bytes) -> str:
    import tempfile
    import subprocess
    import os
    from pathlib import Path
    
    # Create a temporary workspace directory
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_dir = Path(temp_dir) / "workspace"
        workspace_dir.mkdir(exist_ok=True)
        
        # Save the input image to a temporary file
        if image.startswith(b'\x89PNG') or image.startswith(b'\xff\xd8\xff'):
            # PNG or JPEG image
            image_path = workspace_dir / "input_image.png"
            with open(image_path, "wb") as f:
                f.write(image)
        else:
            # Assume PDF
            image_path = workspace_dir / "input_document.pdf" 
            with open(image_path, "wb") as f:
                f.write(image)
        
        # Run olmOCR pipeline
        try:
            cmd = [
                "python", "-m", "olmocr.pipeline", 
                str(workspace_dir),
                "--markdown",
                "--pdfs", str(image_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=temp_dir
            )
            
            print(f"olmOCR command output: {result.stdout}")
            if result.stderr:
                print(f"olmOCR stderr: {result.stderr}")
            
            # Read the markdown output
            markdown_dir = workspace_dir / "markdown"
            if markdown_dir.exists():
                markdown_files = list(markdown_dir.glob("*.md"))
                if markdown_files:
                    output_file = markdown_files[0]
                    with open(output_file, "r", encoding="utf-8") as f:
                        output = f.read()
                    print(f"Result: {output[:500]}...")  # Print first 500 chars
                    return output
                else:
                    return "No markdown output generated"
            else:
                return "Markdown output directory not found"
                
        except subprocess.CalledProcessError as e:
            error_msg = f"olmOCR failed: {e.stderr}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
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
# fn = modal.Function.from_name("example-olmocr-jobs", "parse_receipt")
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
        print(f"running olmOCR on {receipt_filename}")
    else:
        # Use a sample PDF for testing
        receipt_url = "https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running olmOCR on sample PDF from URL {receipt_url}")
    
    result = parse_receipt.remote(image)
    print("=" * 50)
    print("OCR Result:")
    print("=" * 50)
    print(result)
