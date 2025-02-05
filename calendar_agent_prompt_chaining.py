import json
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import os
import logging
from openai import OpenAI
import groq

from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
#client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
#model = "sonar"
model = "llama-3.1-8b-instant"


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
    invitee: str = Field(description="Name of participant to whom the email is addressed")
    body: str = Field(description="Natural language confirmation with event details.")
    calendar_link: Optional[str] = Field(description="Generated calendar link if applicable")


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
                                          f'{{"description": "...", "is_calendar_event": true, "confidence_score": 0.95}}'
             },
            {"role": "user", "content": user_input},
        ],
    )

    str_response = completion.choices[0].message.content
    logger.debug(f"Response: {str_response}")
    response = EventExtraction.model_validate_json(str_response)
    logger.info(
        f"Extraction complete - Is calendar event: {response.is_calendar_event}, Confidence: {response.confidence_score:.2f}"
    )
    logger.info(
        f"Event description: {response.description}"
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
                           f"Return ONLY a valid JSON, without Markdown formatting (` ```json `, ` ``` `), explanations, or extra text. "
                            f"Use the following JSON format: "
                            f'{{"name": "...", "date": "...", "time": "...", "attendees": ["..."]}}'
                f"Make sure the response does not contain any ```json``` or ```json``` formatting. It is very important!"
                f"'date' and 'time' should be in a human-readable format, e.g., 'January 1, 2023' and '2:00 PM', do NOT include day (e.g. 'Friday') in 'date'."
            },
            {"role": "user", "content": description},
        ],
    )

    str_response = completion.choices[0].message.content
    logger.debug(f"Response: {str_response}")
    response = EventDetails.model_validate_json(str_response)

    logger.info(
        f"Parsing complete - Event name: {response.name}, Date: {response.date}, Time: {response.time}, Attendees: {response.attendees}"
    )
    return response

#user_input = "Let's schedule a 2h meeting next Tuesday at 5 pm with Jane to discuss the project."
#event = extract_event_info(user_input=user_input)
#event = EventExtraction(description='Scheduling a 2-hour meeting next Tuesday at 5 pm with Jane to discuss the project', is_calendar_event=True, confidence_score=0.95)
#parse_event_details(event.description)
#parse_event_details("Scheduling a 2-hour meeting next Tuesday at 5 pm with Jane to discuss the project")


def generate_confirmation_email(event: EventDetails) -> EventConfirmationEmail:
    """Generate a confirmation email with the event details."""

    logger.info("Generating confirmation email")
    logger.debug(f"Event details: {event}")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "system",
                "content": f"Generate a natural sounding email message for each of the participants of the event containing the main topic of the meeting. "
                           f"Make sure the email body is similar for all the participants. "
                           f"Limit the email body to 200 words. "
                           f"Sign of with your name; Oksana"
                           f"Only add the 'calendar_link' field if a calendar link is available in the event details, else add None."
                           f"Return ONLY a list of valid JSON objects, without Markdown formatting (` ```json `, ` ``` `), explanations, or extra text. "
                           f"Use the following format: "
                            f'[{{"invitee": "...", "subject": "...", "body": "...", "calendar_link": "..."}}, '
                           f'{{"invitee": "...", "subject": "...", "body": "...", "calendar_link": "..."}}]'
                        f"It is very important that the response does not contain any extra text, just the list of json objects."

            },
            {"role": "user", "content": str(event.model_dump())},
        ],
    )
    response = completion.choices[0].message.content
    logger.debug(f"Response: {response}")
    parsed_response = json.loads(response)
    emails = [EventConfirmationEmail.model_validate(res) for res in parsed_response]
    logger.info(f"Email generated - Subject: {emails[0].subject}")
    return emails

# event = parse_event_details("A meeting with Jane and John to discuss the project roadmap next Tuesday at 2pm")
# generate_confirmation_email(event)

# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------

def process_event(user_input: str) -> EventConfirmationEmail:
    """Process the user input to determine if it's a calendar event, extract event information from user input,
    and generate a confirmation email."""
    logger.info("Processing input")
    logger.debug(f"Raw input: {user_input}")

    # Extract basic event information
    event_info = extract_event_info(user_input)

    # Verify if it's a calendar event with sufficient confidence
    if not event_info.is_calendar_event or event_info.confidence_score < 0.7:
        logger.warning(f"Gate check failed - is_calendar_event: {event_info.is_calendar_event}, confidence: {event_info.confidence_score:.2f}")
        raise ValueError("Not a valid calendar event")

    # Extract event details
    event_details = parse_event_details(event_info.description)

    # Step 3: Generate confirmation email
    confirmation_email = generate_confirmation_email(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation_email

# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

#user_input = "How about scheduling a 90-minute meeting this Friday at 10 AM with Alex and Sarah to review the latest product updates?"
user_input = "Can you send an email to Jane to prepare to discuss the project roadmap?"
result = process_event(user_input)
if result:
    for email in result:
        print(f"Email to {email.invitee}")
        print(f"Subject: {email.subject}")
        print(f"Body: {email.body}")
        if email.calendar_link:
            print(f"Calendar link: {email.calendar_link}")
        print("\n")
else:
    print("This doesn't appear to be a calendar event request.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

#user_input = "Can you send an email to Jane to prepare to discuss the project roadmap?"
