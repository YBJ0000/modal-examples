# PaddleX Integration Issues Summary

## Root Cause Analysis

The majority of issues encountered are related to **dependencies and version compatibility**:
- PaddlePaddle framework versioning issues
- API changes between versions
- GPU compatibility problems
- Container environment configuration

## Resolved Issues

1. **Missing `set_optimization_level` Method**
   - **Problem**: PaddleX called a non-existent method in PaddlePaddle
   - **Solution**: Created a monkey patch that adds a compatible method to `paddle.base.libpaddle.AnalysisConfig`

2. **PaddlePaddle Version Compatibility**
   - **Problem**: Initially tried to use non-existent `paddlepaddle-gpu==3.0.0`
   - **Solution**: Downgraded to available version `paddlepaddle-gpu==2.6.2`

3. **Pip Install Syntax Error**
   - **Problem**: Invalid requirement format when mixing package name with mirror URL
   - **Solution**: Separated package installation from mirror URL specification

4. **File Path Issues**
   - **Problem**: Container couldn't find `/root/tools/download_model.py`
   - **Solution**: Fixed directory copying with `add_local_dir("tools", "/root/tools", copy=True)`

## Unresolved Issues

1. **Segmentation Fault in PaddlePredictor**
   - **Problem**: Crash during variable creation in PaddlePredictor initialization
   - **Error Location**: In C++ code when loading the `PP-DocLayout_plus-L` model
   - **Root Cause**: Likely a deep compatibility issue between PaddlePaddle 2.6.2, PaddleX, and GPU drivers
   - **Attempted Solutions**:
     - Successfully applied the `set_optimization_level` patch
     - Added code to disable TensorRT engine
   - **Potential Next Steps**:
     - Try switching from `device: cuda` to `device: cpu` in model_configs.yaml
     - Consider downgrading PaddleX version
     - Investigate GPU compatibility issues
     - **Alternative Approach**: Replace MonkeyOCR with OlmOCR if these issues persist