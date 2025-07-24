# MonkeyOCR on Modal - Setup Guide

## Project Overview
This project demonstrates how to run MonkeyOCR using Modal's cloud infrastructure for OCR processing of documents and receipts. The application uses PaddlePaddle and MonkeyOCR to perform OCR on images.

## Prerequisites
- macOS, Linux, or Windows with WSL
- Python 3.10+ installed
- Git
- [Modal CLI](https://modal.com/docs/guide/cli) installed and configured
- Conda (recommended for environment management)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/YBJ0000/modal-examples.git
cd modal-examples/09_job_queues
```

### 2. Set Up Python Environment

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment with Python 3.10
conda create -n modal-env python=3.10
conda activate modal-env

# Install Modal
pip install modal

# Log in to Modal
modal token new
```

#### Option B: Using venv
```bash
python -m venv modal-env
source modal-env/bin/activate  # On Windows: modal-env\Scripts\activate

# Install Modal
pip install modal

# Log in to Modal
modal token new
```

### 3. Local Dependencies

While most dependencies are installed in the Modal cloud environment, you need Modal CLI locally:

```bash
pip install modal
```

> **Note**: You don't need to install the heavy dependencies like PaddlePaddle, MonkeyOCR, or PyTorch locally. These are defined in the `inference_image` in `doc_ocr_jobs.py` and will be installed in the Modal cloud environment automatically.

### 4. Run the Application

```bash
modal run doc_ocr_jobs.py
```

This command will:
1. Build a container image with all required dependencies
2. Download the necessary model weights
3. Apply the monkey patch for PaddlePaddle compatibility
4. Run OCR on a sample receipt image (or your provided image)

### 5. Deploy as a Persistent Endpoint (Optional)

To deploy the application as a persistent endpoint:

```bash
modal deploy doc_ocr_jobs.py
```

## Troubleshooting

### Common Issues

1. **PaddlePaddle Version Compatibility**
   - The project currently uses `paddlepaddle-gpu==2.6.2` which is the highest available version
   - If you see version-related errors, check the patch in `patches/monkey_patch.py`

2. **Segmentation Fault**
   - If you encounter a segmentation fault during model loading, try modifying `model_configs.yaml` to use CPU instead of GPU:
     ```yaml
     device: cpu  # Change from 'cuda' to 'cpu'
     ```

3. **File Path Issues**
   - If you see errors about missing files, ensure the local directories are properly copied to the container using `add_local_dir` in `doc_ocr_jobs.py`

4. **Modal Authentication**
   - If you encounter authentication issues, run `modal token new` to refresh your credentials

## Advanced Configuration

### Using Different Models

The project supports both MonkeyOCR and PP-DocLayout_plus-L models. To switch between them, modify the `model_configs.yaml` file:

```yaml
layout_config: 
  model: PP-DocLayout_plus-L  # or doclayout_yolo
```

### Using Your Own Images

To process your own receipt image:

```bash
modal run doc_ocr_jobs.py --receipt_filename path/to/your/receipt.jpg
```

## For Developers

If you need to modify the code:

1. The main OCR function is `parse_receipt` in `doc_ocr_jobs.py`
2. PaddlePaddle compatibility patches are in `patches/monkey_patch.py`
3. Model configuration is in `model_configs.yaml`

After making changes, run the application again with `modal run doc_ocr_jobs.py`.