import os 
import dotenv
from PIL import Image, ImageOps
import numpy as np
import cv2
import io
from openai import OpenAI
import shutil
import base64
from configs import settings
from utils import download_image
import time 

class Inpainter:
    def __init__(self, first_prompt):
        dotenv.load_dotenv()
        self.first_prompt = first_prompt
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("The OPENAI_API_KEY environment variable is not set.")

        self.client = OpenAI()

        # config value setting 
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.inpaint_model = settings['inpaint_model']
        self.img_path = settings['img_path']
        self.inpaint_path = settings['inpaint_path']

    def __call__(self, 
                 mask:dict
                 ):
        return self.inpaint(mask)

    def move_file(self, source_path: str, destination_path: str):
        # Ensure the destination directory exists
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        
        # Move the file
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} to {destination_path}")

    def generate_background(self, prompt):
        """
        args: 
          prompt: str

        return: img_url 
        """

        response = self.client.images.generate(
            prompt=prompt,
            n=1,
            size=self.image_size
        )

        return response.data[0].url
    
    def _inpaint_single(self, image_path, mask_path, prompt):
        """
        args: 
          image: path
          mask: io
          prompt: str

        return: img_url 
        """

        print(image_path, mask_path)

        response = self.client.images.edit(
            model=self.inpaint_model,
            image=open(image_path, "rb"),
            mask=open(mask_path, 'rb'),
            prompt=prompt,
            n=1,
            size=self.image_size 
        )

        return response.data[0].url

    def _draw_background(self, background_prompt):
        background_path = os.path.join(self.img_path, "background.png")

        print(f"backgorund_prompt: {background_prompt}")
        img_url = self.generate_background(background_prompt)
        download_image(img_url, background_path)

        #background_image = Image.new('RGB', (self.image_width, self.image_height), color='white')
        #background_image.save(background_path)
        return background_path

    
    def _create_mask(self, mask:list) -> np.ndarray:
        """
        Creates a mask image based on center coordinates, width, and height.

        Args:
            center_x (int): X coordinate of the rectangle center.
            center_y (int): Y coordinate of the rectangle center.
            width (int): Width of the rectangle.
            height (int): Height of the rectangle.

        Returns:
            np.ndarray: Generated mask image.
        """
        center_x, center_y, width, height = mask
        mask = np.zeros((self.image_width, self.image_height), dtype="uint8")
        top_left_x = center_x - width // 2
        top_left_y = center_y - height // 2
        bottom_right_x = center_x + width // 2
        bottom_right_y = center_y + height // 2
        mask = cv2.rectangle(mask, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), 255, -1)

        return mask
    
    def _make_transparent_mask(self, base_image_path, mask, mask_index):

        base_image = Image.open(base_image_path).convert('RGBA')
        mask_image = Image.fromarray(mask).convert('L')

        # Resize the mask image to match the size of the base image if necessary
        if base_image.size != mask_image.size:
            mask_image = mask_image.resize(base_image.size, Image.LANCZOS)

        # Invert the mask image (white areas will become black, black areas will become white)
        inverted_mask = ImageOps.invert(mask_image)

        # Create an alpha mask from the inverted mask
        alpha_mask = inverted_mask.point(lambda p: p > 128 and 255)

        # Apply the alpha mask to the base image
        base_image.putalpha(alpha_mask)

        # Save the result
        mask_path = f'results/masks/masked_{mask_index}.png'
        base_image.save(mask_path)
        return mask_path

    def _mask_to_image_bytes(self,mask: np.ndarray) -> io.BytesIO:
        """
        Converts a mask (numpy array) to an in-memory image.

        Args:
            mask (np.ndarray): The mask to convert.

        Returns:
            io.BytesIO: The in-memory image.
        """
        # Convert the numpy array to a PIL image
        image = Image.fromarray(mask)
        
        # Save the image to a BytesIO object
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr
    



    # GPT-4o를 사용하여 프롬프트 생성
    def generate_objects_prompt(self, object_name: str, background_prompt) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who generates good prompts for DALL·E2 inpainting."},
                {"role": "user", "content": f'''
                    You should create exact image generation prompt that focuses on ordinary '{object_name}' for inpainting.
                    The background image's prompt is {background_prompt}.
                    It has to match well with the given background prompt.
                    Only give me one short prompt.
                '''}
            ]
        )
        prompt = completion.choices[0].message.content
        return prompt
    
    def copy_file(self, src_file, dest_folder):
        try:
            # 대상 폴더가 없으면 생성
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            
            # 소스 파일명 추출
            file_name = f"{self.first_prompt}_final.png"
            
            # 대상 경로 설정
            dest_file = os.path.join(dest_folder, file_name)
            
            # 파일 복사
            shutil.copy2(src_file, dest_file)
            print(f"Copied: {src_file} to {dest_file}")
            
        except Exception as e:
            print(f"Error: {e}")


    def inpaint(self, mask):
        object_names = mask['object_name']
        num_objects = mask['num_objects']
        position_list = mask['position_list']
        background_prompt = mask['background_prompt']
        
        prev_path = self._draw_background(background_prompt)
        #prev_path = "C:/Users/Noah/Downloads/wdjf.png"
        for idx in range(num_objects):
            object_name = object_names[idx]
            position = position_list[idx]
            file_name = f'inpaint_{self.first_prompt}_{idx}.png'
            current_path = os.path.join(self.inpaint_path, file_name)
            mask = self._create_mask(position)
            mask_path = self._make_transparent_mask(prev_path, mask, idx)
            print(mask_path)
            #prompt = f'inpaint {object_name} on the mask.'
            prompt = self.generate_objects_prompt(object_name, background_prompt)

            print(f"object prompt_{idx}: {prompt}")
            img_url = self._inpaint_single(prev_path, mask_path, prompt)
            download_image(img_url, current_path)
            prev_path = current_path
            #time.sleep(1.5)
        print(prev_path)
        self.copy_file(prev_path, self.img_path)


    
