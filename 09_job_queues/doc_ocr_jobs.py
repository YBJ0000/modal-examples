# ---
# deploy: true
# mypy: ignore-errors
# ---

# # Run a job queue for olmOCR with vLLM acceleration

# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](https://modal.com/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).

# Our job queue will handle a single task: running OCR transcription for images of receipts and documents.
# We'll make use of olmOCR: a toolkit for converting PDFs and other image-based document formats 
# into clean, readable, plain text format using a 7B parameter Vision Language Model.
# This version uses vLLM for accelerated inference.

# ## Define an App

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).
# Later, we'll use the name provided for our `App` to find it from our web app and submit tasks to it.

from typing import Optional
import modal

app = modal.App("example-doc-ocr-jobs")

# We define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).
# olmOCR requires specific dependencies including poppler-utils for PDF processing.
# We also add vLLM for accelerated inference.

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
        "Pillow",
        "PyPDF2"  # 添加PyPDF2用于获取PDF页数
    )
    # Install vLLM for accelerated inference (支持多模态)
    .pip_install("vllm>=0.7.2")
    # Install latest olmOCR with GPU support
    .pip_install("olmocr[gpu]>=0.2.1")
    # Install qwen-vl-utils for easier multimodal handling
    .pip_install("qwen-vl-utils")
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
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    # vLLM性能优化环境变量
    "CUDA_LAUNCH_BLOCKING": "0",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
    "TOKENIZERS_PARALLELISM": "false"
})

# ## Global vLLM engine instance for reuse across requests

vllm_engine = None

def get_vllm_engine():
    """Initialize and return vLLM engine with optimized settings"""
    global vllm_engine
    if vllm_engine is None:
        from vllm import LLM
        print("Initializing vLLM engine...")
        
        # vLLM引擎配置，基于olmOCR官方推荐参数
        vllm_engine = LLM(
            model="allenai/olmOCR-7B-0225-preview",
            trust_remote_code=True,
            dtype="bfloat16",
            # 性能优化参数
            gpu_memory_utilization=0.95,  # 使用95%的GPU内存
            max_model_len=8192,  # 根据需要调整最大序列长度
            tensor_parallel_size=1,  # 单GPU设置
            # 多模态支持
            limit_mm_per_prompt={"image": 1},  # 每个prompt最多1张图片
            max_num_seqs=6,  # 支持批处理
        )
        print("vLLM engine initialized successfully")
    
    return vllm_engine

# ## Run OCR inference on Modal using vLLM acceleration

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator with model caching for optimal performance.

# olmOCR requires a GPU with at least 20GB of GPU RAM. We use A100 which has 40GB.

@app.function(
    gpu="H100",
    retries=3,
    volumes={MODEL_CACHE_PATH: model_cache},
    image=inference_image,
    timeout=600,  # 10 minutes timeout for processing
    # 添加性能优化参数
    keep_warm=1,  # 保持一个实例热启动
    container_idle_timeout=300,  # 5分钟空闲超时
)
def parse_receipt(image: bytes) -> str:
    """
    Optimized olmOCR implementation using vLLM for accelerated inference.
    This version supports multi-page PDF processing.
    """
    import tempfile
    import base64
    from io import BytesIO
    from PIL import Image
    from pathlib import Path
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info
    import PyPDF2
    
    # Import olmOCR modules for preprocessing
    from olmocr.data.renderpdf import render_pdf_to_base64png
    from olmocr.prompts import build_finetuning_prompt
    from olmocr.prompts.anchor import get_anchor_text
    
    # Get the vLLM engine (initialized once and reused)
    llm = get_vllm_engine()
    
    print("Processing document with vLLM-accelerated olmOCR...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Determine file type and save appropriately
            if image.startswith(b'%PDF'):
                # PDF file - process all pages
                pdf_path = temp_path / "input.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(image)
                
                # Get total number of pages
                with open(pdf_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    total_pages = len(pdf_reader.pages)
                
                def process_pdf_batch(pdf_path, total_pages, batch_size=6):
                    """批量处理PDF页面"""
                    llm = get_vllm_engine()
                    all_results = {}
                    
                    # 分批处理
                    for batch_start in range(1, total_pages + 1, batch_size):
                        batch_end = min(batch_start + batch_size - 1, total_pages)
                        batch_pages = list(range(batch_start, batch_end + 1))
                        
                        print(f"Processing batch: pages {batch_pages}")
                        
                        # 准备批量消息
                        batch_messages = []
                        for page_num in batch_pages:
                            try:
                                # 渲染页面
                                image_base64 = render_pdf_to_base64png(
                                    str(pdf_path), 
                                    page_num,
                                    target_longest_image_dim=1024
                                )
                                
                                # 获取锚点文本
                                anchor_text = get_anchor_text(
                                    str(pdf_path), 
                                    page_num,
                                    pdf_engine="pdfreport", 
                                    target_length=4000
                                )
                                
                                # 构建提示
                                prompt = build_finetuning_prompt(anchor_text)
                                
                                # 构建消息
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                                        ],
                                    }
                                ]
                                
                                batch_messages.append(messages)
                                
                            except Exception as e:
                                print(f"Error preparing page {page_num}: {e}")
                                all_results[page_num] = f"[Error preparing page: {e}]"
                        
                        # 批量推理
                        if batch_messages:
                            try:
                                sampling_params = SamplingParams(
                                    max_tokens=2048,
                                    temperature=0.0,
                                    top_p=1.0,
                                    stop=None,
                                )
                                
                                # 关键：一次性处理多个请求
                                outputs = llm.chat(
                                    messages=batch_messages,  # 传入多个messages
                                    sampling_params=sampling_params,
                                    use_tqdm=False
                                )
                                
                                # 处理结果
                                for i, output in enumerate(outputs):
                                    page_num = batch_pages[i]
                                    if output.outputs:
                                        result = output.outputs[0].text.strip()
                                        all_results[page_num] = result
                                        print(f"Page {page_num} processed successfully ({len(result)} characters)")
                                    else:
                                        all_results[page_num] = "[No output generated]"
                                        
                            except Exception as e:
                                print(f"Batch processing error: {e}")
                                for page_num in batch_pages:
                                    all_results[page_num] = f"[Batch error: {e}]"
                    
                    return all_results
                
                # 修改主函数中的PDF处理部分
                # 将原来的for循环替换为：
                if image.startswith(b'%PDF'):
                    # ... PDF准备代码 ...
                    
                    print(f"Processing PDF with {total_pages} pages using batch processing...")
                    
                    # 使用批处理
                    page_results = process_pdf_batch(pdf_path, total_pages, batch_size=6)
                    
                    # 组合结果
                    all_results = []
                    for page_num in range(1, total_pages + 1):
                        if page_num in page_results:
                            all_results.append(f"# Page {page_num}\n\n{page_results[page_num]}")
                        else:
                            all_results.append(f"# Page {page_num}\n\n[Error: Page not processed]")
                    
                    final_result = "\n\n---\n\n".join(all_results)
                    print(f"Batch PDF processing completed. Total output length: {len(final_result)} characters")
                    return final_result
                else:
                    return "Error: No pages could be processed successfully."
                
            elif image.startswith(b'\x89PNG') or image.startswith(b'\xff\xd8\xff'):
                # PNG or JPEG image - single image processing
                pil_image = Image.open(BytesIO(image))
                
                # Convert to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # For images, use minimal anchor text
                anchor_text = ""
                
                # Build the prompt using olmOCR's prompt building system
                prompt = build_finetuning_prompt(anchor_text)
                
                # Build the multimodal message for vLLM
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ],
                    }
                ]
                
                print("Running vLLM inference on image...")
                
                # Configure sampling parameters for OCR (deterministic output)
                sampling_params = SamplingParams(
                    max_tokens=2048,  # Increased for longer documents
                    temperature=0.0,  # Deterministic output for OCR
                    top_p=1.0,
                    stop=None,
                )
                
                # Generate output using vLLM
                outputs = llm.chat(
                    messages=messages,
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                
                # Extract the generated text
                if outputs and len(outputs) > 0:
                    result = outputs[0].outputs[0].text.strip()
                else:
                    result = "No output generated"
                
                print(f"Image OCR processing completed. Output length: {len(result)} characters")
                return result
                
            else:
                return "Error: Unsupported file format. Please provide PDF, PNG, or JPEG files."
            
    except Exception as e:
        error_msg = f"vLLM olmOCR processing failed: {str(e)}"
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
        print(f"running vLLM-accelerated olmOCR on {receipt_filename}")
    else:
        # Use a sample PDF for testing
        receipt_url = "https://olmocr.allenai.org/papers/olmocr_3pg_sample.pdf"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running vLLM-accelerated olmOCR on sample PDF from URL {receipt_url}")
    
    result = parse_receipt.remote(image)
    print("=" * 50)
    print("OCR Result:")
    print("=" * 50)
    print(result)
