import autogen
from typing import List, Dict, Annotated, Tuple
import os
import json
import logging
import dotenv
import cv2
import numpy as np
from configs import settings

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from typing import List



def check_overlap(boxes: List[List[int]]) -> bool:
    def do_boxes_overlap(box1, box2):
        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2

        if left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2:
            return True
        return False

    bounding_boxes = []
    for box in boxes:
        center_x, center_y, width, height = box
        half_width = width // 2
        half_height = height // 2
        top_left_x = center_x - half_width
        top_left_y = center_y - half_height
        bottom_right_x = center_x + half_width
        bottom_right_y = center_y + half_height
        bounding_boxes.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            if do_boxes_overlap(bounding_boxes[i], bounding_boxes[j]):
                return True
    return False

def mask_generation_tool(
        object_name: Annotated[List[str], "Names of each object with personality"],
        num_objects: Annotated[int, "Number of objects"],
        position_list: Annotated[List[List[int]], "Position of each rectangle mask: [center_x, center_y, width, height]"],
        background_prompt: Annotated[str, "prompt for background"],
        output_dir: str = "results/masks"
) -> Dict[str, any]:
    """
    Generates a dictionary containing object names, number of objects, and positions of rectangle masks.

    Args:
        object_name (List[str]): Name of each object with personality.
        num_objects (int): Number of rectangle masks.
        position_list (List[List[int]]): List of positions and sizes for each rectangle mask.
        background_prompt: prompt for background"
        output_dir (str): Directory where the masks will be saved (not used here).

    Returns:
        Dict[str, any]: Dictionary containing object names, number of objects, and positions of masks.
    """

    # Validate input
    if len(position_list) != num_objects:
        error_msg = "The length of position_list does not match num_objects."
        logging.error(error_msg)
        return {"error": error_msg}

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory at {output_dir}")

    masks = []
    for i in range(num_objects):
        try:
            center_x, center_y, width, height = position_list[i]
            mask = (center_x, center_y, width, height)
            masks.append({
                "object_name": object_name[i],
                "mask": mask
            })
        except Exception as e:
            error_msg = f"Error generating mask for {object_name[i]}_{i}: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

    result = {
        "object_name": object_name,
        "num_objects": num_objects,
        "position_list": position_list,
        "background_prompt": background_prompt
    }

    json_output_path = os.path.join(output_dir, 'masks_data.json')
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(result, json_file)
        logging.info(f"Data saved to {json_output_path}")
    except Exception as e:
        error_msg = f"Error saving data to JSON file: {str(e)}"
        logging.error(error_msg)
        return error_msg

    return "Great! All data is saved to masks_data.json. TERMINATE"





class agents:
    def __init__(self):
        
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.model_name = settings['mask_gen_model']

        # LLM configuration

        config_list = autogen.config_list_from_json(
            env_or_file="configs/OAI_CONFIG_LIST_4.json",
            filter_dict={"model": [self.model_name]}
        )

        seed = 12
        llm_config = {
            "cache_seed": seed,
            "temperature": 0,
            "config_list": config_list,
            "timeout": 120,
        }
        self.llm_config = llm_config


        self.background_bot = autogen.AssistantAgent(
            name="background_bot",
            system_message=(
                f"Your task is to create a short background prompt based on the user's initial prompt. Ensure the background matches the given prompt but does not contain any objects, only scenery. When all tasks of all agents are completed, print TERMINATE. Provide the 'background_prompt' (str) to the mask_generation_bot."
                "The background should be consistent, with all sky if it's supposed to be the sky, and all ground if it's supposed to be the ground, and all sea if it's supposed to be the sea. "
                #"You are responsible for generating a short background prompt based on the user's first prompt. "
                #"The backgounrd prompt must have a background that matches the given prompt."
                #"The background prompt must not contains any obejcts. just sight"
                #"When all processes of ALL agents are completed, print TERMINATE."
                #"Provide the 'background_prompt' (str) to the mask_generation_bot. "
            ),
            llm_config=self.llm_config,
        )




        self.position_bot = autogen.AssistantAgent(
            name="position_bot",
            system_message=(
                f"Your role is to determine the positions and sizes of rectangles within an image of size {self.image_size}. Ensure the rectangles stay within the image boundaries. Make each rectangle's height and width more large. Make sure the rectangles must be sufficiently spaced apart. Position the objects naturally within the scene described by the user prompt and background prompt. Provide the 'position_list' (List[List[int]]) to the position_verifier_bot. Format each item in the list as [center_x, center_y, width, height]. All object names must be in a single list. Combine all object names and their positions into a single list before sending it to the position_verifier_bot."

                #f"You are responsible for determining the positions and sizes of rectangles surely within an image of size {self.image_size}. "
                #"Ensure the rectangles surely do not overlap and stay within the image boundaries. Provide the 'position_list' (List[List[int]]) to the position_verifier_bot. "
                #"Each item in the list should be formatted as [center_x, center_y, width, height]. "
                #"Make sure the bounding boxes do not overlap or go beyond the given image size boundaries. Adjust positions and sizes always to fit within the boundaries. "
                #"All object names must be included in a single list. "
                #"Ensure that all rectangles are positioned as far apart from each other as possible."
                #"If necessary, make reasonable guesses to position the objects naturally within the scene described by the user prompt. "
                #"Combine all object names and their positions into a single list before sending it to the position_verifier_bot."
            ),
            llm_config=self.llm_config,
        )


        self.position_verifier_bot = autogen.AssistantAgent(
            name="position_verifier_bot",
            system_message=(
                f"Your task is to verify the correctness and naturalness of the positions and sizes of the rectangles based on their names and coordinates. The images are of size {self.image_size}. Each bounding box should be formatted as (object name, [center_x, center_y, width, height]). Ensure the bounding boxes stay within the image boundaries. Adjust positions and sizes if necessary. Ensure all bounding boxes are sufficiently spaced apart. If any overlap occurs, request repositioning from the position_bot. Verify all object names are included in a single list and the positions fit within the {self.image_size} image. If correct, provide the combined position_list to the mask_generation_bot. If not, request repositioning from the position_bot with reasons for adjustments."

                #"You are responsible for verifying the correctness and naturalness of the positions and sizes of the rectangles based on their names and given coordinates. "
                #f"The images are of size {self.image_size}. Each bounding box should be in the format (object name, [center_x, center_y, width, height]). "
                #"Make sure the bounding boxes do not overlap or go beyond the given image size boundaries. Adjust positions and sizes if necessary to fit within the boundaries. "
                #"Ensure that all bounding boxes do not overlap and are sufficiently spaced apart. "
                #"If any bounding boxes overlap, request repositioning from the position_bot. "
                #f"All object names must be included in a single list, and verify that the positions and sizes do not exceed the boundaries of a {self.image_size} image. "
                #"Combine all object names and their positions into a single list before sending it to the mask_generation_bot."
                #"If the positions are correct, provide the combined position_list to the mask_generation_bot. "
                #"If the positions are not correct, make position_bot to repositioning and give the reason."
                #"small width and height are not good"
            ),
            llm_config=self.llm_config
        )



        self.mask_generation_bot = autogen.AssistantAgent(
            name="mask_generation_bot",
            system_message=(
                f"Your task is to generate the masks based on the verified positions and sizes of the rectangles, ensuring they align with their names and given coordinates. The images are of size {self.image_size}. Each bounding box should be formatted as (object name, [center_x, center_y, width, height]). Ensure the bounding boxes do not overlap or go beyond the image boundaries. Adjust positions and sizes if necessary. If any overlap or exceed the image size, request repositioning from the position_bot. When all agents have completed their tasks, print TERMINATE."

                #"You are responsible for verifying the correctness and naturalness of the positions and sizes of the rectangles based on their names and given coordinates. "
                #f"The images are of size {self.image_size}. Each bounding box should be in the format (object name, [center_x, center_y, width, height]). "
                #"Make sure the bounding boxes do not overlap or go beyond the given image size boundaries. Adjust positions and sizes if necessary to fit within the boundaries. "
                #"Ensure that all bounding boxes do not overlap and are sufficiently spaced apart. "
                #"If any bounding boxes overlap or exceed the image size, request repositioning from the position_bot. "
                #"When all processes of ALL agents are completed, print TERMINATE."

            ),
            llm_config=self.llm_config
        )



        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=8,
            code_execution_config=False
        )

        self.group_chat = autogen.GroupChat(
            agents=[self.background_bot, self.user_proxy, self.position_bot, self.mask_generation_bot, self.position_verifier_bot],
            messages=[],
            max_round=20
        )
        self.manager = autogen.GroupChatManager(groupchat=self.group_chat, llm_config=self.llm_config)

        self._register_tools()


    def _register_tools(self):
        self.user_proxy.register_for_execution(self.initiate_chat)

        self.mask_generation_bot.register_for_llm(name="mask_generator", description="mask generation tool")(mask_generation_tool)
        self.user_proxy.register_for_execution(name="mask_generator")(mask_generation_tool)

        self.position_verifier_bot.register_for_llm(name="check_overlap", description="Check if bounding boxes overlap")(check_overlap)
        self.user_proxy.register_for_execution(name="check_overlap")(check_overlap)

    def initiate_chat(self, first_message: str) -> None:
        self.user_proxy.initiate_chat(
            self.manager,
            message=first_message
        )



# Example usage
if __name__ == "__main__":
    

    generator = agents()
    prompt = "Draw three balls "
    
    result = generator.initiate_chat(prompt)


    print(result)

    print(type(result))
