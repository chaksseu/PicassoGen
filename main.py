import argparse
import json

from models import agents, Inpainter

def main() :
    parser = argparse.ArgumentParser(description="parser setting")

    parser.add_argument('--prompt', required=True, type=str, help='prompt')

    args = parser.parse_args()
    first_prompt = args.prompt

    prompt =f"Make rectangle masks for the following prompt: '{first_prompt}'. " +  f"At first The background_bot make prompt for background with given prompt: {first_prompt}. And give it to position_bot and mask_generation_bot. Next, The position_bot will decide the positions and sizes first, and the position_verifier_bot will verify the positions. " + "If there are no issues, the mask_generation_bot will generate and save the masks. If there are problems, the position_bot must reposition."


    ai_agents = agents()
    inpainter = Inpainter(first_prompt)

    ai_agents.initiate_chat(prompt)
    mask_dict_path = 'results/masks/masks_data.json'

    with open(mask_dict_path, 'r') as f:
        data = json.load(f)

    inpainter(data)

if __name__ == '__main__':
    main()
