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
                print(f"Handling set_optimization_level({level}) - this method is not directly supported in this version")
                # 这里我们不调用tensorrt_optimization_level，因为它是getter而不是setter
                # 相反，我们只是记录调用并返回self以支持链式调用
                return self
            
            # 添加方法到类
            setattr(AnalysisConfig, 'set_optimization_level', set_optimization_level)
            print("Successfully patched AnalysisConfig.set_optimization_level")
    except Exception as e:
        print(f"Failed to patch paddle predictor: {e}")

# 执行补丁
patch_paddle_predictor()