import json
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import os
import logging
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
model = "sonar"

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------

class EventExtraction(BaseModel):
    """ Extract basic event information """
    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(description="Whether this text describes a calendar event")
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """ Parse specific event details """
    name: str = Field(description="Name of the event")
    date: str = Field(description="Date of the event")
    time: str = Field(description="Time of the event")
    attendees: list[str] = Field(description="List of attendees")


class EventConfirmationEmail(BaseModel):
    """ Generate email response with generated event details """
    subject: str = Field(description="Subject of the email")
    body: str = Field(description="Natural language confirmation with event details.")
    link: Optional[str] = Field(description="Generated calendar link if applicable")


# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------

def extract_event_info(user_input: str) -> EventExtraction:
    """Extract basic event information from the input text."""

    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f"{date_context} Analyze if the text describes a calendar event. "
                                          f"Return only a valid JSON, without Markdown formatting (` ```json `, ` ``` `), explanations, or extra text. "
                                          f"Use the following JSON format: "
                                          f'{{"description": "...", "is_calendar_event": True, "confidence_score": 0.95}}'
             },
            {"role": "user", "content": user_input},
        ],
    )

    str_response = completion.choices[0].message.content
    response =EventExtraction.model_validate_json(str_response)
    logger.info(
        f"Extraction complete - Is calendar event: {response.is_calendar_event}, Confidence: {response.confidence_score:.2f}"
    )
    return response


# user_input = "Let's schedule a 2h meeting next Tuesday at 2pm with Jane and John to discuss the project roadmap."
# extract_event_info(user_input=user_input)


def parse_event_details(description: EventExtraction) -> EventDetails:
    """Parse specific event details from the extracted information."""

    logger.info("Parsing event details")
    logger.debug(f"Event info: {description}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"{date_context} Extract detailed event information. "
                           f"When dates reference 'next Tuesday' or similar relative dates, use this current date as reference."
                           f"Return only a valid JSON, without Markdown formatting (` ```json `, ` ``` `), explanations, or extra text. "
                            f"Use the following JSON format: "
                            f'{{"name": "...", "date": "...", "time": "...", "attendees": ["..."]}}'
                f'"date" and "time" values should be formatted to ISO 8601 format'
            },
            {"role": "user", "content": description},
        ],
    )

    str_response = completion.choices[0].message.content
    response = EventDetails.model_validate_json(str_response)
    logger.info(
        f"Parsing complete - Event name: {response.name}, Date: {response.date}, Time: {response.time}, Attendees: {response.attendees}"
    )
    return response

#user_input = "Let's schedule a 2h meeting next Tuesday at 5 pm with Jane to discuss the project."
#event = extract_event_info(user_input=user_input)
#event = EventExtraction(description='Scheduling a 2-hour meeting next Tuesday at 5 pm with Jane to discuss the project', is_calendar_event=True, confidence_score=0.95)
#parse_event_details(event.description)
parse_event_details("Scheduling a 2-hour meeting next Tuesday at 5 pm with Jane to discuss the project")


# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------


# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = "Let's schedule a 2h meeting next Tuesday at 2pm with Jane and John to discuss the project roadmap."

# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

user_input = "Can you send an email to Jane to prepare to discuss the project roadmap?"
