import modal

app = modal.App("test-dependencies")

# 创建测试镜像
test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "wget",
        "curl",
        "git",
        "build-essential",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender-dev",
        "libgomp1",
        "libgcc-s1"
    )
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
        "opencv-contrib-python",
        "pypdfium2"
    )
    .run_commands([
        # 先卸载可能存在的旧版本
        "python -m pip uninstall -y paddlepaddle paddlepaddle-gpu paddlex paddleocr || true",
        # 尝试安装 PaddlePaddle 3.0 预发布版本
        "python -m pip install --pre --upgrade paddlepaddle-gpu || echo 'PaddlePaddle 3.0 预发布版本安装失败，回退到稳定版本'",
        # 如果预发布版本失败，安装稳定版本
        "python -c 'import paddle; print(f\"当前 PaddlePaddle 版本: {paddle.__version__}\")' || python -m pip install paddlepaddle-gpu",
        # 验证最终安装的版本
        "python -c 'import paddle; print(f\"最终 PaddlePaddle 版本: {paddle.__version__}\")'",
        # 安装其他组件
        "python -m pip install paddlex paddleocr",
        "python -m pip install git+https://github.com/Yuliang-Liu/MonkeyOCR.git"
    ])
    .add_local_dir("tools", "/root/tools", copy=True)
    .run_commands([
        "python -m pip install huggingface_hub",
        "python /root/tools/download_model.py -n MonkeyOCR-pro-1.2B"
    ])
    .add_local_file("model_configs.yaml", "/root/model_configs.yaml")
)

@app.function(image=test_image, timeout=300, retries=0)  # 设置 retries=0 避免循环
def test_all_dependencies():
    """安全测试所有依赖，避免段错误导致的循环"""
    import sys
    import os
    
    def test_import(module_name, package_name=None):
        try:
            if package_name:
                __import__(package_name)
            else:
                __import__(module_name)
            print(f"✅ {module_name} - 导入成功")
            return True
        except Exception as e:
            print(f"❌ {module_name} - 导入失败: {e}")
            return False
    
    print("🔍 开始安全测试所有依赖...")
    
    # 基础依赖测试
    dependencies = [
        ("torch", None),
        ("torchvision", None),
        ("numpy", None),
        ("PIL", None),
        ("cv2", None),
        ("paddle", None),
        ("paddlex", None),
        ("transformers", None),
        ("yaml", None),
        ("dill", None),
    ]
    
    failed = []
    for module, package in dependencies:
        if not test_import(module, package):
            failed.append(module)
    
    # 测试 PaddlePaddle 版本
    try:
        import paddle
        version = paddle.__version__
        print(f"✅ PaddlePaddle 版本: {version}")
        
        # 检查版本
        version_parts = version.split('.')
        major_version = int(version_parts[0])
        
        if major_version >= 3:
            print(f"🎉 成功安装 PaddlePaddle 3.x: {version}")
            compatibility_mode = False
        else:
            print(f"⚠️  使用 PaddlePaddle 2.x: {version}，需要兼容性处理")
            compatibility_mode = True
            
    except Exception as e:
        print(f"❌ PaddlePaddle 版本检查失败: {e}")
        failed.append("paddle_version")
        compatibility_mode = True
    
    # 安全测试 paddle.inference.Config
    try:
        import paddle
        config = paddle.inference.Config()
        print("✅ paddle.inference.Config 创建成功")
        
        # 检查关键方法
        if hasattr(config, 'set_optimization_level'):
            print("✅ set_optimization_level 方法存在")
        else:
            print("⚠️  set_optimization_level 方法不存在，需要兼容性处理")
            
        if hasattr(config, 'set_tensorrt_optimization_level'):
            print("✅ set_tensorrt_optimization_level 方法存在")
            
    except Exception as e:
        print(f"❌ paddle.inference.Config 测试失败: {e}")
        failed.append("paddle_inference")
    
    # 安全测试 MonkeyOCR 导入（不初始化）
    try:
        from magic_pdf.model.custom_model import MonkeyOCR
        print("✅ MonkeyOCR 模块导入成功")
        
        # 检查类是否存在
        if MonkeyOCR:
            print("✅ MonkeyOCR 类定义正常")
        
        # 不进行实际初始化，避免段错误
        print("⚠️  跳过 MonkeyOCR 初始化测试以避免段错误")
        
    except Exception as e:
        print(f"❌ MonkeyOCR 导入失败: {e}")
        failed.append("MonkeyOCR")
    
    # 总结
    if failed:
        print(f"\n❌ 失败的模块: {failed}")
        print("\n📋 建议的解决方案:")
        if "paddle" in failed or "paddle_version" in failed:
            print("1. PaddlePaddle 安装问题 - 需要检查安装源和版本")
        if "MonkeyOCR" in failed:
            print("2. MonkeyOCR 兼容性问题 - 可能需要修改源码或使用兼容性补丁")
        if compatibility_mode:
            print("3. 当前使用 PaddlePaddle 2.x，建议升级到 3.x 或使用兼容性补丁")
        return False
    else:
        print("\n✅ 基础依赖测试通过!")
        if compatibility_mode:
            print("⚠️  注意：使用 PaddlePaddle 2.x，MonkeyOCR 可能需要兼容性处理")
        return True

@app.local_entrypoint()
def main():
    try:
        result = test_all_dependencies.remote()
        if result:
            print("🎉 基础依赖测试通过!")
            print("💡 下一步：需要为 MonkeyOCR 创建兼容性补丁或升级 PaddlePaddle")
        else:
            print("💥 依赖测试失败，需要修复配置")
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")