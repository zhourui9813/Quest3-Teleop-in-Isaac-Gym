import pygame
import numpy as np
import threading
import time

class PygameDisplay:
    """
    Pygame显示器类，用于调试可视化
    避免使用全局变量，提供更好的代码组织
    """
    
    def __init__(self, width=2560, height=720, title="Teleop Debug Display", fps=30):
        """
        初始化pygame显示器
        
        Args:
            width: 显示窗口宽度
            height: 显示窗口高度  
            title: 窗口标题
            fps: 显示帧率限制
        """
        self.width = width
        self.height = height
        self.title = title
        self.fps = fps
        
        # 显示相关变量
        self.screen = None
        self.clock = None
        self.font = None
        self.initialized = False
        
        # 性能优化：控制显示频率
        self.last_display_time = 0
        self.display_interval = 1.0 / fps  # 显示间隔
        
        # 线程锁，确保线程安全
        self.lock = threading.Lock()
        
    def initialize(self):
        """初始化pygame显示"""
        with self.lock:
            if not self.initialized:
                pygame.init()
                pygame.display.set_caption(self.title)
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 36)
                self.initialized = True
                print(f"Pygame display initialized: {self.width}x{self.height}")
    
    def should_update_display(self):
        """检查是否应该更新显示（基于帧率限制）"""
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            return True
        return False
    
    def display_image(self, img_left, img_right, title="Debug Display"):
        """
        显示左右图像
        
        Args:
            img_left: 左侧图像 (numpy array)
            img_right: 右侧图像 (numpy array)
            title: 显示标题
        """
        # 性能优化：限制显示频率
        if not self.should_update_display():
            return
            
        if not self.initialized:
            self.initialize()
        
        with self.lock:
            # 处理pygame事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    return
            
            # 合并左右图像
            combined_img = np.hstack((img_left, img_right))
            
            # 转换图像格式 (numpy array -> pygame surface)
            # pygame需要的格式是 (width, height, 3)，但numpy是 (height, width, 3)
            img_surface = pygame.surfarray.make_surface(combined_img.swapaxes(0, 1))
            
            # 清空屏幕
            self.screen.fill((0, 0, 0))
            
            # 显示图像
            self.screen.blit(img_surface, (0, 0))
            
            # 添加标题文本
            text_surface = self.font.render(title, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
            # 添加帧率信息
            fps_text = f"FPS: {self.clock.get_fps():.1f}"
            fps_surface = self.font.render(fps_text, True, (255, 255, 0))
            self.screen.blit(fps_surface, (10, 50))
            
            # 更新显示
            pygame.display.flip()
            self.clock.tick(self.fps)
    
    def cleanup(self):
        """清理pygame资源"""
        with self.lock:
            if self.initialized:
                pygame.quit()
                self.initialized = False
                print("Pygame display cleaned up")
    
    def is_initialized(self):
        """检查是否已初始化"""
        return self.initialized
    
    def __del__(self):
        """析构函数，确保资源清理"""
        self.cleanup()

# 可选：提供一个简单的全局实例（如果需要的话）
_global_display = None

def get_global_display():
    """获取全局显示器实例（单例模式）"""
    global _global_display
    if _global_display is None:
        _global_display = PygameDisplay()
    return _global_display

def cleanup_global_display():
    """清理全局显示器"""
    global _global_display
    if _global_display is not None:
        _global_display.cleanup()
        _global_display = None