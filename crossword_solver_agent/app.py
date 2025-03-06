import os

import yaml

from smolagents import CodeAgent, HfApiModel, tool
from dotenv import load_dotenv
from PIL import Image
from tools.web_search import DuckDuckGoSearchTool
import json

import re
import numpy as np
import pytesseract
from io import BytesIO

load_dotenv()  # Load environment variables from .env file
# Use environment variable
os.environ.get("HF_TOKEN")  # Remove the hardcoded token line

from Gradio_UI import GradioUI


# Define the model
model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)



# Read the prompts from the prompts.yaml file
with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)


search_tool = DuckDuckGoSearchTool()

@tool
def ocr_tool(crossword_image: Image.Image) -> str:
    """
    Extracts crossword clues from an image using OCR.
    
    Args:
        crossword_image: An image containing a crossword puzzle with clues.
        
    Returns:
        A JSON string containing the extracted clues in the format:
        [{"direction":"across", "number":"1", "question":"Deli offering"}, ...]
    """
    # Ensure pytesseract knows where tesseract is installed
    # pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract' # Uncomment and set this if needed
    
    # Convert image to numpy array for processing
    img_np = np.array(crossword_image)
    
    # Extract text from image
    extracted_text = pytesseract.image_to_string(img_np)
    
    # Process the extracted text to identify clues
    clues = []
    
    # Pattern to match clues like "1. Across: Deli offering" or "1. Down: Movie theater snack"
    across_pattern = r'(\d+)[.\s]+[Aa]cross:?\s+(.*?)(?=\d+[.\s]+|$)'
    down_pattern = r'(\d+)[.\s]+[Dd]own:?\s+(.*?)(?=\d+[.\s]+|$)'
    
    # Alternative pattern if clues are arranged differently
    alt_pattern = r'(\d+)[.\s]+(.*?)(?=\d+[.\s]+|$)'
    
    # Extract across clues
    across_matches = re.findall(across_pattern, extracted_text)
    for number, question in across_matches:
        clues.append({
            "direction": "across",
            "number": number.strip(),
            "question": question.strip()
        })
    
    # Extract down clues
    down_matches = re.findall(down_pattern, extracted_text)
    for number, question in down_matches:
        clues.append({
            "direction": "down",
            "number": number.strip(),
            "question": question.strip()
        })
    
    # If the standard patterns don't work, try the alternative pattern
    # and use heuristics to determine direction
    if not clues:
        sections = extracted_text.lower().split("across")
        if len(sections) > 1:
            across_section = sections[1].split("down")[0] if "down" in sections[1] else sections[1]
            down_section = sections[1].split("down")[1] if "down" in sections[1] else ""
            
            # Process across section
            across_matches = re.findall(alt_pattern, across_section)
            for number, question in across_matches:
                clues.append({
                    "direction": "across",
                    "number": number.strip(),
                    "question": question.strip()
                })
            
            # Process down section
            down_matches = re.findall(alt_pattern, down_section)
            for number, question in down_matches:
                clues.append({
                    "direction": "down",
                    "number": number.strip(),
                    "question": question.strip()
                })
    
    # Also extract the grid structure
    # This is a simplified approach - in a real implementation, 
    # you might need more sophisticated image processing
    
    # Return the clues as a JSON string
    return json.dumps(clues, indent=2)


@tool
def crossword_solver_tool(clues_json: str) -> str:
    """
    Solves crossword clues using an LLM and search_tool to find the answer if needed.
    
    Args:
        clues_json: A JSON string containing crossword clues in the format:
                   [{"direction":"across", "number":"1", "question":"Deli offering"}, ...]
                   
    Returns:
        A JSON string with the original clues and their answers in the format:
        [{"direction":"across", "number":"1", "question":"Deli offering", "answer":["REUBEN", "SANDWICH"]}, ...]
    """
    clues = json.loads(clues_json)
    # Create a prompt for the LLM to solve all clues at once
    prompt = "Solve the following New Yorker style crossword clues."\
        "Provide only the answer for each clue in ALL CAPS:\n\n."\
        "If more than one answer possible, provide all answers in a list separated by commas. Use the search_tool to find the answer if needed. "
    
    for i, clue in enumerate(clues):
        prompt += f"{i+1}. {clue['direction'].capitalize()} {clue['number']}: {clue['question']}\n"
    
    # Get solutions from the LLM
    response = model.complete(prompt)
    
    # Parse the response to extract answers
    # This is a simplified approach - you might need more robust parsing
    answer_pattern = r'(\d+)\.\s+.*?:\s+([A-Z]+)'
    answers = re.findall(answer_pattern, response)
    
    # Create a dictionary to store answers by their index
    answer_dict = {int(idx): ans for idx, ans in answers}
    
    # Add answers to the original clues
    for i, clue in enumerate(clues):
        if i+1 in answer_dict:
            clue["answer"] = answer_dict[i+1]
        else:
            # If no answer was found, provide a placeholder
            clue["answer"] = "UNKNOWN"
    
    # Return the updated clues with answers as a JSON string
    return json.dumps(clues, indent=2)


@tool
def format_solutions_tool(solved_clues_json: str) -> str:
    """
    Formats the solved crossword clues into a readable Markdown format.
    
    Args:
        solved_clues_json: A JSON string containing solved clues in the format:
                          [{"direction":"across", "number":"1", "question":"Deli offering", "answer":"REUBEN"}, ...]
                   
    Returns:
        A formatted Markdown string with the crossword solutions.
    """
    clues = json.loads(solved_clues_json)
    
    # Separate clues into across and down
    across_clues = [c for c in clues if c["direction"].lower() == "across"]
    down_clues = [c for c in clues if c["direction"].lower() == "down"]
    
    # Sort clues by number
    across_clues.sort(key=lambda x: int(x["number"]))
    down_clues.sort(key=lambda x: int(x["number"]))
    
    # Build markdown output
    markdown = "# Crossword Solutions\n\n"
    
    # Format across clues
    markdown += "## Across\n\n"
    for clue in across_clues:
        number = clue["number"]
        question = clue["question"]
        answer = clue.get("answer", "UNKNOWN")
        if isinstance(answer, list):
            answer = " / ".join(answer)  # Handle multiple possible answers
        markdown += f"**{number}**. {question} → **{answer}**\n\n"
    
    # Format down clues
    markdown += "## Down\n\n"
    for clue in down_clues:
        number = clue["number"]
        question = clue["question"]
        answer = clue.get("answer", "UNKNOWN")
        if isinstance(answer, list):
            answer = " / ".join(answer)  # Handle multiple possible answers
        markdown += f"**{number}**. {question} → **{answer}**\n\n"
    
    return markdown


# Update the agent tools list
agent = CodeAgent(
    model=model,
    tools=[search_tool, ocr_tool, crossword_solver_tool, format_solutions_tool],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()









# @tool
# def meme_generator(meme_description:str)-> Image:
#     """
#     This tool creates a meme image according to a prompt, which is a text description.
#     Args:
#         meme_description: A string representing the description of the meme image you want to create.
#     Returns:
#         An image of the meme.
#     """
#     inputs = {"prompt": {"type": "string", 
#                          "description": "Describe the meme image you want to create, including style and any text elements."}}
    
#     # Add style instructions to the prompt
#     styled_prompt = f"Create a meme in the style of the New Yorker magazine, with a cartoonish style. The image should be: {meme_description}"
#     # OR for pixel art style:
#     # styled_prompt = f"Create a pixel art style meme with the following elements: {description}"
#     # OR for watercolor style:
#     # styled_prompt = f"Create a watercolor painting style meme of: {description}"
    

#     model_sdxl = "runwayml/stable-diffusion-v1-5" #"black-forest-labs/FLUX.1-schnell"
#     client = InferenceClient(model_sdxl)
#     image = client.text_to_image(styled_prompt)
#     return image





# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 



# # Import tool from Hub
# image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
# search_tool = DuckDuckGoSearchTool()


    


