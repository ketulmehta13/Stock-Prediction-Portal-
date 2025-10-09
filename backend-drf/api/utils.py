import os
from django.conf import settings
import matplotlib.pyplot as plt

def save_plot(plot_img_path):
    # Ensure media directory exists
    os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
    
    # Create full image path
    image_path = os.path.join(settings.MEDIA_ROOT, plot_img_path)
    
    # Save the plot
    plt.savefig(image_path, bbox_inches='tight', dpi=100)
    plt.close()
    
    # Return full URL (including domain for development)
    image_url = f'http://127.0.0.1:8000{settings.MEDIA_URL}{plot_img_path}'
    return image_url
