# 修复PaddleX中的set_optimization_level错误
import importlib
import sys

def patch_paddle_predictor():
    try:
        # 尝试导入paddle.base.libpaddle.AnalysisConfig
        from paddle.base.libpaddle import AnalysisConfig
        
        # 保存原始方法
        original_set_optimization_level = getattr(AnalysisConfig, 'set_optimization_level', None)
        
        # 如果方法不存在，添加兼容方法
        if original_set_optimization_level is None:
            def set_optimization_level(self, level):
                print(f"Handling set_optimization_level({level}) - PaddlePaddle 2.6.2 compatibility mode")
                
                # 尝试禁用TensorRT引擎，可能会避免一些兼容性问题
                try:
                    if hasattr(self, 'disable_tensorrt_engine'):
                        self.disable_tensorrt_engine()
                        print("Disabled TensorRT engine")
                except Exception as e:
                    print(f"Failed to disable TensorRT engine: {e}")
                
                # 简单返回self以支持链式调用
                return self
            
            # 添加方法到类
            setattr(AnalysisConfig, 'set_optimization_level', set_optimization_level)
            print("Successfully patched AnalysisConfig.set_optimization_level for PaddlePaddle 2.6.2")
    except Exception as e:
        print(f"Failed to patch paddle predictor: {e}")

# 执行补丁
patch_paddle_predictor()