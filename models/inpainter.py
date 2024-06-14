import os
import dotenv
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
from openai import OpenAI
import shutil
from configs import settings
from utils import download_image

class Inpainter:
    def __init__(self, first_prompt):
        # Load environment variables
        dotenv.load_dotenv()
        
        # Initialize attributes
        self.first_prompt = first_prompt
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Check if the API key is set
        if not api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")
        
        # Initialize OpenAI client
        self.client = OpenAI()
        
        # Set image dimensions and model settings
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.inpaint_model = settings['inpaint_model']
        self.img_path = settings['img_path']
        self.inpaint_path = settings['inpaint_path']

    def __call__(self, mask: dict):
        return self.inpaint(mask)

    def move_file(self, source_path: str, destination_path: str):
        # Ensure destination directory exists and move file
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} to {destination_path}")

    def generate_background(self, prompt: str) -> str:
        # Generate a background image based on the prompt
        response = self.client.images.generate(
            prompt=prompt,
            n=1,
            size=self.image_size
        )
        return response.data[0].url

    def _inpaint_single(self, image_path: str, mask_path: str, prompt: str) -> str:
        # Perform inpainting on a single image section
        response = self.client.images.edit(
            model=self.inpaint_model,
            image=open(image_path, "rb"),
            mask=open(mask_path, 'rb'),
            prompt=prompt,
            n=1,
            size=self.image_size
        )
        return response.data[0].url

    def _draw_background(self, background_prompt: str) -> str:
        # Draw the background based on the prompt
        background_path = os.path.join(self.img_path, "background.png")
        img_url = self.generate_background(background_prompt)
        download_image(img_url, background_path)
        return background_path

    def _create_mask(self, mask: list) -> np.ndarray:
        # Create a mask based on the provided coordinates
        center_x, center_y, width, height = mask
        mask_img = np.zeros((self.image_width, self.image_height), dtype="uint8")
        top_left_x = center_x - width // 2
        top_left_y = center_y - height // 2
        bottom_right_x = center_x + width // 2
        bottom_right_y = center_y + height // 2
        cv2.rectangle(mask_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)
        return mask_img

    def _make_transparent_mask(self, base_image_path: str, mask: np.ndarray, mask_index: int) -> str:
        # Make a mask with transparency
        base_image = Image.open(base_image_path).convert('RGBA')
        mask_image = Image.fromarray(mask).convert('L')
        
        # Resize the mask if necessary
        if base_image.size != mask_image.size:
            mask_image = mask_image.resize(base_image.size, Image.LANCZOS)
        
        inverted_mask = ImageOps.invert(mask_image)
        alpha_mask = inverted_mask.point(lambda p: p > 128 and 255)
        base_image.putalpha(alpha_mask)
        
        # Save the mask image
        mask_path = f'results/masks/masked_{mask_index}.png'
        base_image.save(mask_path)
        return mask_path

    def _mask_to_image_bytes(self, mask: np.ndarray) -> io.BytesIO:
        # Convert mask to image bytes for storage or processing
        image = Image.fromarray(mask)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def generate_objects_prompt(self, object_name: str, background_prompt: str) -> str:
        # Generate a prompt for inpainting objects
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who generates good prompts for DALL·E2 inpainting."},
                {"role": "user", "content": f"Create an image generation prompt focusing on an ordinary '{object_name}' for inpainting. The background image's prompt is {background_prompt}. Ensure it matches well with the given background prompt. Provide only one short prompt."}
            ]
        )
        return completion.choices[0].message.content

    def copy_file(self, src_file: str, dest_folder: str):
        # Copy a file to the destination folder
        os.makedirs(dest_folder, exist_ok=True)
        file_name = f"{self.first_prompt}_final.png"
        dest_file = os.path.join(dest_folder, file_name)
        shutil.copy2(src_file, dest_file)
        print(f"Copied: {src_file} to {dest_file}")

    def inpaint(self, mask: dict):
        # Main inpainting function
        object_names = mask['object_name']
        num_objects = mask['num_objects']
        position_list = mask['position_list']
        background_prompt = mask['background_prompt']

        # Generate the initial background
        prev_path = self._draw_background(background_prompt)
        
        # Process each object to inpaint
        for idx in range(num_objects):
            object_name = object_names[idx]
            position = position_list[idx]
            file_name = f'inpaint_{self.first_prompt}_{idx}.png'
            current_path = os.path.join(self.inpaint_path, file_name)
            mask_img = self._create_mask(position)
            mask_path = self._make_transparent_mask(prev_path, mask_img, idx)
            prompt = self.generate_objects_prompt(object_name, background_prompt)
            img_url = self._inpaint_single(prev_path, mask_path, prompt)
            download_image(img_url, current_path)
            prev_path = current_path

        # Copy the final inpainted image to the destination
        self.copy_file(prev_path, self.img_path)
