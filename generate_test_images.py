#!/usr/bin/env python3
"""
Generate various 2-channel test images for testing the CNN model
"""
import numpy as np
from PIL import Image
import os

def create_test_images():
    """Create different types of 2-channel test images"""
    
    # Create test_images directory
    os.makedirs('test_images', exist_ok=True)
    
    # 1. Grayscale + Alpha (circle with transparency)
    print("Creating grayscale + alpha image...")
    img_size = 128
    
    # Create a circle with gradient
    y, x = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    
    # Channel 1: Grayscale circle with gradient
    distance = np.sqrt((x - center)**2 + (y - center)**2)
    circle_mask = distance <= 40
    gradient = np.clip(255 - distance * 3, 0, 255).astype(np.uint8)
    grayscale = gradient * circle_mask
    
    # Channel 2: Alpha (transparency) - fade out towards edges
    alpha = np.clip(255 - distance * 4, 0, 255).astype(np.uint8)
    alpha = alpha * circle_mask + 50 * (~circle_mask)  # semi-transparent background
    
    # Combine channels
    la_array = np.stack([grayscale, alpha], axis=-1).astype(np.uint8)
    la_image = Image.fromarray(la_array, mode='LA')
    la_image.save('test_images/circle_grayscale_alpha.png')
    
    # 2. Depth + Confidence simulation
    print("Creating depth + confidence image...")
    
    # Channel 1: Simulated depth map (closer = brighter)
    depth_map = np.zeros((img_size, img_size), dtype=np.uint8)
    for i in range(img_size):
        for j in range(img_size):
            # Create a 3D surface effect
            depth_val = int(127 + 60 * np.sin(i/10) * np.cos(j/10))
            depth_map[i, j] = np.clip(depth_val, 0, 255)
    
    # Channel 2: Confidence map (center more confident)
    confidence = 255 - distance.astype(np.uint8)
    confidence = np.clip(confidence, 50, 255)
    
    # Combine channels
    depth_conf_array = np.stack([depth_map, confidence], axis=-1).astype(np.uint8)
    depth_conf_image = Image.fromarray(depth_conf_array, mode='LA')
    depth_conf_image.save('test_images/depth_confidence.png')
    
    # 3. Two different patterns
    print("Creating dual pattern image...")
    
    # Channel 1: Horizontal stripes
    stripes_h = np.tile(np.linspace(0, 255, img_size//8, dtype=np.uint8), (img_size//8, 1))
    stripes_h = np.tile(stripes_h, (8, 8))[:img_size, :img_size]
    
    # Channel 2: Vertical stripes
    stripes_v = np.tile(np.linspace(0, 255, img_size//8, dtype=np.uint8).reshape(-1, 1), (1, img_size//8))
    stripes_v = np.tile(stripes_v, (8, 8))[:img_size, :img_size]
    
    # Combine channels
    dual_pattern_array = np.stack([stripes_h, stripes_v], axis=-1).astype(np.uint8)
    dual_pattern_image = Image.fromarray(dual_pattern_array, mode='LA')
    dual_pattern_image.save('test_images/dual_patterns.png')
    
    # 4. Noise + Signal
    print("Creating noise + signal image...")
    
    # Channel 1: Random noise
    noise = np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)
    
    # Channel 2: Clean geometric pattern
    signal = np.zeros((img_size, img_size), dtype=np.uint8)
    # Create concentric squares
    for i in range(5):
        start = i * 15
        end = img_size - i * 15
        if start < end:
            signal[start:end, start:end] = 50 + i * 40
    
    # Combine channels
    noise_signal_array = np.stack([noise, signal], axis=-1).astype(np.uint8)
    noise_signal_image = Image.fromarray(noise_signal_array, mode='LA')
    noise_signal_image.save('test_images/noise_signal.png')
    
    # 5. Motion vectors simulation
    print("Creating motion vectors image...")
    
    # Channel 1: X-direction motion (horizontal flow)
    x_motion = np.sin(np.linspace(0, 4*np.pi, img_size)).reshape(-1, 1)
    x_motion = np.tile(x_motion, (1, img_size))
    x_motion = ((x_motion + 1) * 127.5).astype(np.uint8)
    
    # Channel 2: Y-direction motion (vertical flow)
    y_motion = np.cos(np.linspace(0, 4*np.pi, img_size)).reshape(1, -1)
    y_motion = np.tile(y_motion, (img_size, 1))
    y_motion = ((y_motion + 1) * 127.5).astype(np.uint8)
    
    # Combine channels
    motion_array = np.stack([x_motion, y_motion], axis=-1).astype(np.uint8)
    motion_image = Image.fromarray(motion_array, mode='LA')
    motion_image.save('test_images/motion_vectors.png')
    
    print("\n✅ Created 5 test images in 'test_images/' folder:")
    print("1. circle_grayscale_alpha.png - Grayscale circle with alpha transparency")
    print("2. depth_confidence.png - Simulated depth map with confidence")
    print("3. dual_patterns.png - Horizontal and vertical stripe patterns")
    print("4. noise_signal.png - Random noise + geometric pattern")
    print("5. motion_vectors.png - Simulated X/Y motion vectors")
    print("\nYou can upload these to test your CNN model!")

if __name__ == "__main__":
    create_test_images()