import argparse
import json
from models import Agents, Inpainter

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="parser setting")
    
    # Add prompt argument which is required
    parser.add_argument('--prompt', required=True, type=str, help='prompt')
    args = parser.parse_args()
    
    # Get the prompt from the arguments
    first_prompt = args.prompt

    # Construct a detailed prompt for the AI agents
    prompt = f'''
    Make rectangle masks for the following prompt: '{first_prompt}'.
    First, the background_bot will create a prompt for the background based on the given prompt: {first_prompt}.
    It will then pass this to the position_bot and mask_generation_bot.
    Next, the position_bot will decide the positions and sizes, and the position_verifier_bot will verify them.
    If there are no issues, the mask_generation_bot will generate and save the masks.
    If there are problems, the position_bot must reposition.
    '''

    # Initialize AI agents and inpainter with the given prompt
    ai_agents = Agents()
    inpainter = Inpainter(first_prompt)

    # Initiate chat with the AI agents using the constructed prompt
    ai_agents.initiate_chat(prompt)
    
    # Path to the JSON file containing mask data
    mask_dict_path = 'results/masks/masks_data.json'

    # Load the mask data from the JSON file
    with open(mask_dict_path, 'r') as f:
        data = json.load(f)

    # Call the inpainter with the loaded mask data
    inpainter(data)

if __name__ == '__main__':
    main()
