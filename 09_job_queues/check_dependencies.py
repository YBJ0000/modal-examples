#!/usr/bin/env python3
"""
本地依赖检查脚本 - 快速验证 MonkeyOCR 所需的所有依赖
在本地运行此脚本可以快速发现缺失的依赖，避免在 Modal 上反复试错
"""

import sys
import subprocess
import importlib.util

def check_import(module_name, package_name=None):
    """检查模块是否可以导入"""
    try:
        if package_name:
            __import__(package_name)
        else:
            __import__(module_name)
        print(f"✅ {module_name} - 导入成功")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - 导入失败: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name} - 导入时出现其他错误: {e}")
        return False

def check_paddle_version():
    """检查 PaddlePaddle 版本兼容性"""
    try:
        import paddle
        version = paddle.__version__
        print(f"✅ paddle 版本: {version}")
        
        # 检查版本是否为 3.0+
        major_version = int(version.split('.')[0])
        if major_version >= 3:
            print("✅ PaddlePaddle 版本兼容 (3.0+)")
        else:
            print(f"❌ PaddlePaddle 版本过低 ({version})，需要 3.0+")
            return False
            
        # 检查 inference 配置
        if hasattr(paddle, 'inference'):
            config = paddle.inference.Config()
            if hasattr(config, 'set_optimization_level'):
                print("✅ paddle.inference.Config.set_optimization_level - 方法存在")
            else:
                print("❌ paddle.inference.Config.set_optimization_level - 方法不存在")
                return False
        
        return True
    except Exception as e:
        print(f"❌ paddle 检查失败: {e}")
        return False

def main():
    print("🔍 开始检查 MonkeyOCR 相关依赖...")
    print("=" * 50)
    
    # 基础依赖检查
    dependencies = [
        ("torch", None),
        ("torchvision", None), 
        ("numpy", None),
        ("PIL", "Pillow"),
        ("cv2", "opencv-contrib-python"),
        ("paddle", "paddlepaddle-gpu>=3.0.0"),
        ("paddlex", None),
        ("transformers", None),
        ("yaml", "PyYAML"),
        ("dill", None),
    ]
    
    failed_imports = []
    
    for module, package in dependencies:
        if not check_import(module, package):
            failed_imports.append((module, package or module))
    
    print("\n" + "=" * 50)
    
    # 特殊检查：PaddlePaddle 版本
    print("🔧 检查 PaddlePaddle 版本兼容性...")
    if not check_paddle_version():
        failed_imports.append(("paddle", "paddlepaddle-gpu>=3.0.0"))
    
    print("\n" + "=" * 50)
    
    # 尝试导入 MonkeyOCR
    print("🐒 检查 MonkeyOCR 导入...")
    try:
        from magic_pdf.model.custom_model import MonkeyOCR
        print("✅ MonkeyOCR 导入成功")
    except Exception as e:
        print(f"❌ MonkeyOCR 导入失败: {e}")
        failed_imports.append(("MonkeyOCR", "git+https://github.com/Yuliang-Liu/MonkeyOCR.git"))
    
    print("\n" + "=" * 50)
    
    if failed_imports:
        print("❌ 发现缺失或不兼容的依赖:")
        for module, package in failed_imports:
            print(f"   - {package}")
        print("\n建议的 pip install 命令:")
        packages = [package for _, package in failed_imports]
        print(f"pip install {' '.join(packages)}")
    else:
        print("✅ 所有依赖检查通过!")
    
    return len(failed_imports) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)