import pygame
import numpy as np
import threading
import time

class PygameDisplay:
    """
    Pygame display class for debugging visualization
    Avoids using global variables, provides better code organization
    """
    
    def __init__(self, width=2560, height=720, title="Teleop Debug Display", fps=30):
        """
        Initialize pygame display
        
        Args:
            width: Display window width
            height: Display window height  
            title: Window title
            fps: Display frame rate limit
        """
        self.width = width
        self.height = height
        self.title = title
        self.fps = fps
        
        # Display-related variables
        self.screen = None
        self.clock = None
        self.font = None
        self.initialized = False
        
        # Performance optimization: control display frequency
        self.last_display_time = 0
        self.display_interval = 1.0 / fps  # Display interval
        
        # Thread lock to ensure thread safety
        self.lock = threading.Lock()
        
    def initialize(self):
        """Initialize pygame display"""
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
        """Check if display should be updated (based on frame rate limit)"""
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            return True
        return False
    
    def display_image(self, img_left, img_right, title="Debug Display"):
        """
        Display left and right images
        
        Args:
            img_left: Left image (numpy array)
            img_right: Right image (numpy array)
            title: Display title
        """
        # Performance optimization: limit display frequency
        if not self.should_update_display():
            return
            
        if not self.initialized:
            self.initialize()
        
        with self.lock:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.cleanup()
                    return
            
            # Combine left and right images
            combined_img = np.hstack((img_left, img_right))
            
            # Convert image format (numpy array -> pygame surface)
            # pygame requires format (width, height, 3), but numpy is (height, width, 3)
            img_surface = pygame.surfarray.make_surface(combined_img.swapaxes(0, 1))
            
            # Clear screen
            self.screen.fill((0, 0, 0))
            
            # Display image
            self.screen.blit(img_surface, (0, 0))
            
            # Add title text
            text_surface = self.font.render(title, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
            
            # Add frame rate information
            fps_text = f"FPS: {self.clock.get_fps():.1f}"
            fps_surface = self.font.render(fps_text, True, (255, 255, 0))
            self.screen.blit(fps_surface, (10, 50))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)
    
    def cleanup(self):
        """Clean up pygame resources"""
        with self.lock:
            if self.initialized:
                pygame.quit()
                self.initialized = False
                print("Pygame display cleaned up")
    
    def is_initialized(self):
        """Check if initialized"""
        return self.initialized
    
    def __del__(self):
        """Destructor to ensure resource cleanup"""
        self.cleanup()

# Optional: provide a simple global instance (if needed)
_global_display = None

def get_global_display():
    """Get global display instance (singleton pattern)"""
    global _global_display
    if _global_display is None:
        _global_display = PygameDisplay()
    return _global_display

def cleanup_global_display():
    """Clean up global display"""
    global _global_display
    if _global_display is not None:
        _global_display.cleanup()
        _global_display = None