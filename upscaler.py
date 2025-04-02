import argparse
import os
from realesrgan import RealESRGAN
from PIL import Image

def upscale_image(input_path, output_path, scale=4, model='RealESRGAN_x4plus'):  
    # Load model
    model = RealESRGAN(model)
    model.load_model()
    
    # Open image
    image = Image.open(input_path).convert("RGB")
    
    # Perform upscaling
    upscaled_image = model.enhance(image, scale=scale)
    
    # Save output
    upscaled_image.save(output_path)
    print(f"Upscaled image saved at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upscale images using Real-ESRGAN.")
    parser.add_argument("input", type=str, help="Path to input image.")
    parser.add_argument("output", type=str, help="Path to save upscaled image.")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling factor (default: 4x).")
    parser.add_argument("--model", type=str, default="RealESRGAN_x4plus", help="Model to use for upscaling.")
    
    args = parser.parse_args()
    upscale_image(args.input, args.output, args.scale, args.model)
