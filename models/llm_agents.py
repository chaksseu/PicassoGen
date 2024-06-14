import autogen
from typing import List, Dict, Annotated
import os
import json
import logging
import dotenv
from configs import settings

# Load environment variables from a .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_overlap(boxes: List[List[int]]) -> bool:
    """
    Check if any of the given bounding boxes overlap.
    
    Args:
        boxes (List[List[int]]): List of bounding boxes, each defined by center_x, center_y, width, and height.
        
    Returns:
        bool: True if any boxes overlap, False otherwise.
    """
    def do_boxes_overlap(box1, box2):
        # Determine if two boxes overlap based on their coordinates
        left1, top1, right1, bottom1 = box1
        left2, top2, right2, bottom2 = box2
        return left1 < right2 and right1 > left2 and top1 < bottom2 and bottom1 > top2

    # Convert center coordinates and sizes to bounding box edges
    bounding_boxes = [
        (center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2)
        for center_x, center_y, width, height in boxes
    ]

    # Check each pair of boxes for overlap
    for i in range(len(bounding_boxes)):
        for j in range(i + 1, len(bounding_boxes)):
            if do_boxes_overlap(bounding_boxes[i], bounding_boxes[j]):
                return True
    return False

def mask_generation_tool(
    object_name: Annotated[List[str], "Names of each object with personality"],
    num_objects: Annotated[int, "Number of objects"],
    position_list: Annotated[List[List[int]], "Position of each rectangle mask: [center_x, center_y, width, height]"],
    background_prompt: Annotated[str, "Prompt for background"],
    output_dir: str = "results/masks"
) -> Dict[str, any]:
    """
    Generates a dictionary containing object names, number of objects, and positions of rectangle masks.

    Args:
        object_name (List[str]): Names of each object.
        num_objects (int): Number of objects.
        position_list (List[List[int]]): Positions of each rectangle mask.
        background_prompt (str): Prompt for the background.
        output_dir (str): Directory to save the output. Default is "results/masks".
        
    Returns:
        Dict[str, any]: Dictionary containing generated mask information or an error message.
    """
    # Validate that the number of positions matches the number of objects
    if len(position_list) != num_objects:
        error_msg = "The length of position_list does not match num_objects."
        logging.error(error_msg)
        return {"error": error_msg}

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Ensured output directory exists at {output_dir}")

    masks = []
    # Generate masks for each object
    for i in range(num_objects):
        try:
            mask = position_list[i]
            masks.append({"object_name": object_name[i], "mask": mask})
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

    # Save the result to a JSON file
    json_output_path = os.path.join(output_dir, 'masks_data.json')
    try:
        with open(json_output_path, 'w') as json_file:
            json.dump(result, json_file)
        logging.info(f"Data saved to {json_output_path}")
    except Exception as e:
        error_msg = f"Error saving data to JSON file: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}

    return "Great! All data is saved to masks_data.json. TERMINATE"

class Agents:
    def __init__(self):
        # Initialize agent configuration
        self.image_height = settings['image_height']
        self.image_width = settings['image_width']
        self.image_size = f'{self.image_width}x{self.image_height}'
        self.model_name = settings['llm_model']

        # Load configuration for the model
        config_list = autogen.config_list_from_json(
            env_or_file="configs/OAI_CONFIG_LIST_4o.json",
            filter_dict={"model": [self.model_name]}
        )

        self.llm_config = {
            "cache_seed": 12,
            "temperature": 0,
            "config_list": config_list,
            "timeout": 120,
        }

        # Initialize background bot agent
        self.background_bot = autogen.AssistantAgent(
            name="background_bot",
            system_message=(
                '''
                Your task is to create a short background prompt based on the user's initial prompt.
                Ensure the background matches the given prompt but does not contain any objects, only scenery.
                When all tasks of all agents are completed, print TERMINATE.
                Provide the 'background_prompt' (str) to the mask_generation_bot.
                '''
            ),
            llm_config=self.llm_config,
        )

        # Initialize position bot agent
        self.position_bot = autogen.AssistantAgent(
            name="position_bot",
            system_message=(
                f'''
                Your role is to determine the positions and sizes of rectangles within an image of size {self.image_size}.
                Ensure the rectangles stay within the image boundaries.
                Make each rectangle's height and width larger.
                Ensure the rectangles are sufficiently spaced apart.
                Position the objects naturally within the scene described by the user prompt and background prompt.
                Provide the 'position_list' (List[List[int]]) to the position_verifier_bot.
                Format each item in the list as [center_x, center_y, width, height].
                All object names must be in a single list.
                Combine all object names and their positions into a single list before sending it to the position_verifier_bot.
                '''
            ),
            llm_config=self.llm_config,
        )

        # Initialize position verifier bot agent
        self.position_verifier_bot = autogen.AssistantAgent(
            name="position_verifier_bot",
            system_message=(
                f'''
                Your task is to verify the correctness and naturalness of the positions and sizes of the rectangles based on their names and coordinates.
                The images are of size {self.image_size}.
                Each bounding box should be formatted as (object name, [center_x, center_y, width, height]).
                Ensure the bounding boxes stay within the image boundaries.
                Adjust positions and sizes if necessary.
                Ensure all bounding boxes are sufficiently spaced apart.
                If any overlap occurs, request repositioning from the position_bot.
                Verify all object names are included in a single list and the positions fit within the {self.image_size} image.
                If correct, provide the combined position_list to the mask_generation_bot.
                If not, request repositioning from the position_bot with reasons for adjustments.
                '''        
            ),
            llm_config=self.llm_config
        )

        # Initialize mask generation bot agent
        self.mask_generation_bot = autogen.AssistantAgent(
            name="mask_generation_bot",
            system_message=(
                f'''
                Your task is to generate the masks based on the verified positions and sizes of the rectangles, ensuring they align with their names and given coordinates.
                The images are of size {self.image_size}.
                Each bounding box should be formatted as (object name, [center_x, center_y, width, height]).
                Ensure the bounding boxes do not overlap or go beyond the image boundaries.
                Adjust positions and sizes if necessary.
                If any overlap or exceed the image size, request repositioning from the position_bot.
                When all agents have completed their tasks, print TERMINATE.
                '''
            ),
            llm_config=self.llm_config
        )

        # Initialize user proxy agent
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=8,
            code_execution_config=False
        )

        # Initialize group chat with agents
        self.group_chat = autogen.GroupChat(
            agents=[
                self.background_bot, 
                self.user_proxy, 
                self.position_bot, 
                self.mask_generation_bot, 
                self.position_verifier_bot
            ],
            messages=[],
            max_round=15
        )
        self.manager = autogen.GroupChatManager(groupchat=self.group_chat, llm_config=self.llm_config)
        self._register_tools()

    def _register_tools(self):
        # Register functions as tools for agents
        self.user_proxy.register_for_execution(self.initiate_chat)
        self.mask_generation_bot.register_for_llm(name="mask_generator", description="mask generation tool")(mask_generation_tool)
        self.user_proxy.register_for_execution(name="mask_generator")(mask_generation_tool)
        self.position_verifier_bot.register_for_llm(name="check_overlap", description="Check if bounding boxes overlap")(check_overlap)
        self.user_proxy.register_for_execution(name="check_overlap")(check_overlap)

    def initiate_chat(self, first_message: str) -> None:
        """
        Initiates the chat with the user proxy agent.

        Args:
            first_message (str): The initial message to start the chat.
        """
        self.user_proxy.initiate_chat(self.manager, message=first_message)

# Example usage
if __name__ == "__main__":
    generator = Agents()
    prompt = "Draw three balls"
    result = generator.initiate_chat(prompt)
    print(result)
    print(type(result))
