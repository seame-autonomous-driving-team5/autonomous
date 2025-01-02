from PIL import Image

# Function to display image size
def show_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"Image Dimensions: {width}x{height} (Width x Height)")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
image_path = "blackbox.jpg"  # Replace with your image file path
show_image_size(image_path)