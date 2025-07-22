# ---
# deploy: true
# mypy: ignore-errors
# ---

# # Run a job queue for GOT-OCR

# This tutorial shows you how to use Modal as an infinitely scalable job queue
# that can service async tasks from a web app. For the purpose of this tutorial,
# we've also built a [React + FastAPI web app on Modal](https://modal.com/docs/examples/doc_ocr_webapp)
# that works together with it, but note that you don't need a web app running on Modal
# to use this pattern. You can submit async tasks to Modal from any Python
# application (for example, a regular Django app running on Kubernetes).

# Our job queue will handle a single task: running OCR transcription for images of receipts.
# We'll make use of a pre-trained model:
# the [General OCR Theory (GOT) 2.0 model](https://huggingface.co/stepfun-ai/GOT-OCR2_0).

# Try it out for yourself [here](https://modal-labs-examples--example-doc-ocr-webapp-wrapper.modal.run/).

# [![Webapp frontend](https://modal-cdn.com/doc_ocr_frontend.jpg)](https://modal-labs-examples--example-doc-ocr-webapp-wrapper.modal.run/)

# ## Define an App

# Let's first import `modal` and define an [`App`](https://modal.com/docs/reference/modal.App).
# Later, we'll use the name provided for our `App` to find it from our web app and submit tasks to it.

from typing import Optional

import modal

app = modal.App("example-doc-ocr-jobs")

# We also define the dependencies for our Function by specifying an
# [Image](https://modal.com/docs/guide/images).

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "numpy>=1.21.6,<2.0.0",
        "pydantic>=2.7.2",
        "PyMuPDF>=1.24.9,<=1.24.14",
        "pdfminer.six==20231228",
        "pycocotools>=2.0.6",
        "transformers==4.52.4",
        "qwen_vl_utils==0.0.10",
        "matplotlib",
        "doclayout_yolo==0.0.2b1",
        "PyYAML",
        "dill>=0.3.8,<1",
        "pdf2image==1.17.0",
        "openai==1.88.0",
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "Pillow",
        "paddlex",
        "pypdfium2",
        "opencv-contrib-python",
        "git+https://github.com/Yuliang-Liu/MonkeyOCR.git"
    )
    .add_local_dir("tools", "/root/tools", copy=True)
    .run_commands([
        "python -m pip install huggingface_hub",
        "python /root/tools/download_model.py -n MonkeyOCR-pro-1.2B"
    ])
    .add_local_file("model_configs.yaml", "/root/model_configs.yaml")
)

# ## Run OCR inference on Modal by wrapping with `app.function`

# Now let's set up the actual OCR inference.

# Using the [`@app.function`](https://modal.com/docs/reference/modal.App#function)
# decorator, we set up a Modal [Function](https://modal.com/docs/reference/modal.Function).
# We provide arguments to that decorator to customize the hardware, scaling, and other features
# of the Function.

# Here, we say that this Function should use NVIDIA L40S [GPUs](https://modal.com/docs/guide/gpu),
# automatically [retry](https://modal.com/docs/guide/retries#function-retries) failures up to 3 times,
# and have access to our [shared model cache](https://modal.com/docs/guide/volumes).


# 直接用 monkeyocr 推理
@app.function(
    gpu="l40s",
    retries=3,
    image=inference_image,
)
def parse_receipt(image: bytes) -> str:
    from tempfile import NamedTemporaryFile
    from magic_pdf.model.custom_model import MonkeyOCR
    from PIL import Image
    from importlib.util import find_spec

    config_path = "/root/model_configs.yaml"
    model = MonkeyOCR(config_path)

    # 将 image bytes 保存为临时图片
    with NamedTemporaryFile(delete=False, mode="wb+") as temp_img_file:
        temp_img_file.write(image)
        temp_img_file.flush()
        temp_img_file.seek(0)
        img = Image.open(temp_img_file.name)
        # 这里假设 monkeyocr 有 chat_model.batch_inference 方法，参考 parse.py
        instruction = "Please output the text content from the image."
        result = model.chat_model.batch_inference([img], [instruction])
        output = result[0] if result else ""

    print("Result: ", output)
    return output


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
# then scale back down to 0 when there's no work left. To see how you could use this from a Python web
# app, take a look at the [receipt parser frontend](https://modal.com/docs/examples/doc_ocr_webapp)
# tutorial.

# ## Run manually

# We can also trigger `parse_receipt` manually for easier debugging:

# ```shell
# modal run doc_ocr_jobs
# ```

# To try it out, you can find some
# example receipts [here](https://drive.google.com/drive/folders/1S2D1gXd4YIft4a5wDtW99jfl38e85ouW).


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
        print(f"running OCR on {receipt_filename}")
    else:
        receipt_url = "https://modal-cdn.com/cdnbot/Brandys-walmart-receipt-8g68_a_hk_f9c25fce.webp"
        request = urllib.request.Request(receipt_url)
        with urllib.request.urlopen(request) as response:
            image = response.read()
        print(f"running OCR on sample from URL {receipt_url}")
    print(parse_receipt.remote(image))
