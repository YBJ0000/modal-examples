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
            # 添加一个属性来存储优化级别
            setattr(AnalysisConfig, '_optimization_level', 0)
            
            def set_optimization_level(self, level):
                print(f"Handling set_optimization_level({level}) - storing value")
                # 存储优化级别
                self._optimization_level = level
                
                # 如果有tensorrt_engine_enabled方法，确保它被调用
                if hasattr(self, 'tensorrt_engine_enabled') and callable(getattr(self, 'tensorrt_engine_enabled')):
                    self.tensorrt_engine_enabled()
                    
                # 如果有enable_tensorrt_engine方法，尝试调用它
                if hasattr(self, 'enable_tensorrt_engine') and callable(getattr(self, 'enable_tensorrt_engine')):
                    try:
                        self.enable_tensorrt_engine()
                    except Exception as e:
                        print(f"Warning: Failed to call enable_tensorrt_engine: {e}")
                
                return self
            
            # 添加方法到类
            setattr(AnalysisConfig, 'set_optimization_level', set_optimization_level)
            print("Successfully patched AnalysisConfig.set_optimization_level with enhanced version")
    except Exception as e:
        print(f"Failed to patch paddle predictor: {e}")

# 执行补丁
patch_paddle_predictor()