import os
from smolagents import CodeAgent,DuckDuckGoSearchTool, HfApiModel,load_tool,tool
from tools.web_search import DuckDuckGoSearchTool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from dotenv import load_dotenv
from PIL import Image
from huggingface_hub import InferenceClient
import json
from bs4 import BeautifulSoup
import re

load_dotenv()  # Load environment variables from .env file
# Use environment variable
os.environ.get("HF_TOKEN")  # Remove the hardcoded token line

from Gradio_UI import GradioUI

# Below is an example of a tool that does nothing. Amaze us with your creativity !
@tool
def meme_generator(meme_description:str)-> Image:
    """
    This tool creates a meme image according to a prompt, which is a text description.
    Args:
        meme_description: A string representing the description of the meme image you want to create.
    Returns:
        An image of the meme.
    """
    inputs = {"prompt": {"type": "string", 
                         "description": "Describe the meme image you want to create, including style and any text elements."}}
    
    # Add style instructions to the prompt
    styled_prompt = f"Create a meme in the style of the New Yorker magazine, with a cartoonish style. The image should be: {meme_description}"
    # OR for pixel art style:
    # styled_prompt = f"Create a pixel art style meme with the following elements: {description}"
    # OR for watercolor style:
    # styled_prompt = f"Create a watercolor painting style meme of: {description}"
    

    model_sdxl = "runwayml/stable-diffusion-v1-5" #"black-forest-labs/FLUX.1-schnell"
    client = InferenceClient(model_sdxl)
    image = client.text_to_image(styled_prompt)
    return image


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()

# If the agent does not answer, the model is overloaded, please use another model or the following Hugging Face Endpoint that also contains qwen2.5 coder:
# model_id='https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud' 

model = HfApiModel(
max_tokens=2096,
temperature=0.5,
model_id='Qwen/Qwen2.5-Coder-32B-Instruct',# it is possible that this model may be overloaded
custom_role_conversions=None,
)


# Import tool from Hub
image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)
search_tool = DuckDuckGoSearchTool()

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
@tool
def generate_flashcards(url: str) -> str:
    """
    Creates flashcards from Hugging Face course content at the given URL.
    Generates question-answer pairs based on course sections and their content.
    
    Args:
        url: URL of the Hugging Face course page to parse
        
    Returns:
        str: Path to the generated flashcards JSON file
    """
    try:
        # Verify it's a Hugging Face course URL
        if not url.startswith("https://huggingface.co/learn/"):
            return "Error: URL must be a Hugging Face course page"
            
        # Fetch webpage content
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        flashcards = []
        
        # Find all section headings and their content
        for section in soup.find_all(['h1', 'h2', 'h3']):
            section_title = section.get_text(strip=True)
            
            # Skip navigation or utility sections
            skip_sections = ['Table of contents', 'Navigation', 'Menu', 'Search']
            if any(skip in section_title for skip in skip_sections):
                continue
                
            # Get content until next heading
            content = []
            next_elem = section.find_next_sibling()
            while next_elem and next_elem.name not in ['h1', 'h2', 'h3']:
                if next_elem.name in ['p', 'li', 'code']:
                    text = next_elem.get_text(strip=True)
                    if text:  # Only add non-empty text
                        content.append(text)
                next_elem = next_elem.find_next_sibling()
            
            if content:
                # Create different types of questions based on the content
                
                content_text = " ".join(content)
                print(content_text)
                prompt = f"""Given this section titled "{section_title}" with the following content:

{content_text}

Generate five different quiz questions with their answers based on the key information in this content. 
Format each Q&A pair as a JSON object with 'question' and 'answer' fields.
Keep the answers concise but informative."""
                try:
                    response = model.complete(prompt)
                    # Extract JSON objects from the response using regex
                    json_matches = re.findall(r'\{[^{}]*\}', response)
                    
                    for json_str in json_matches:
                        try:
                            qa_pair = json.loads(json_str)
                            if 'question' in qa_pair and 'answer' in qa_pair:
                                flashcards.append(qa_pair)
                        except json.JSONDecodeError:
                            continue
                            
                except Exception as e:
                    # Fallback to basic question if LLM generation fails
                    flashcards.append({
                        "question": f"What is {section_title}?",
                        "answer": " ".join(content[:2])
                    })

                # Basic "What is" question
                # flashcards.append({
                #     "question": f"What is {section_title}?",
                #     "answer": " ".join(content[:2])  # Use first two paragraphs for concise answer
                # })
                
                # # If section has bullet points, create list-based questions
                # if any('•' in c or '-' in c for c in content):
                #     list_items = [c for c in content if c.strip().startswith(('•', '-'))]
                #     if list_items:
                #         flashcards.append({
                #             "question": f"List the key points about {section_title}",
                #             "answer": "\n".join(f"- {item.strip('•- ')}" for item in list_items)
                #         })
                
                # # If content mentions prerequisites or requirements
                # if any(keyword in ' '.join(content).lower() for keyword in ['prerequisite', 'require', 'need']):
                #     flashcards.append({
                #         "question": f"What are the prerequisites or requirements for {section_title}?",
                #         "answer": next((c for c in content if any(keyword in c.lower() 
                #                     for keyword in ['prerequisite', 'require', 'need'])), "")
                #     })
        
        # Generate filename with course name and timestamp
        course_name = url.rstrip('/').split('/')[-1]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flashcards_{course_name}_{timestamp}.json"
        
        # Save flashcards to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "source_url": url,
                    "generated_at": timestamp,
                    "course": course_name
                },
                "flashcards": flashcards
            }, f, indent=4, ensure_ascii=False)
            
        return filename
        
    except Exception as e:
        return f"Error generating flashcards: {str(e)}"

# Update the agent tools list
agent = CodeAgent(
    model=model,
    tools=[final_answer, search_tool, meme_generator, generate_flashcards],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()