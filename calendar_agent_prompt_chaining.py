import json
import pickle
import sys
from typing import Optional
from datetime import datetime, timezone

from google_auth_httplib2 import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pydantic import BaseModel, Field, field_validator
import os
import logging
from openai import OpenAI
import groq

from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
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
    calendar_link: Optional[str] = Field(description="Generated calendar link if applicable")


class EventConfirmationEmail(BaseModel):
    """ Generate email response with generated event details """
    subject: str = Field(description="Subject of the email")
    invitee: str = Field(description="Name of participant to whom the email is addressed")
    body: str = Field(description="Natural language confirmation with event details.")



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
                          f"Only add the 'calendar_link' field if a calendar link is available in the event details, else add None."
                           f"Return ONLY a valid JSON, without Markdown formatting (` ```json `, ` ``` `), explanations, or extra text. "
                            f"Use the following JSON format: "
                            f'{{"name": "...", "date": "...", "time": "...", "attendees": ["..."], "calendar_link": "..."}}'
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
        f"Parsing complete - Event name: {response.name}, Date: {response.date}, Time: {response.time}, Attendees: {response.attendees}, Calendar link: {response.calendar_link}"
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
                "content": f"Generate a natural sounding email message for each of the participants of the event. "
                           f"If there are no participant's DON'T generate anything and return an empty list."
                           f"Match the style of the conversation with original input style. "
                           f"Make sure the email body is similar for all the participants. "
                           f"Limit the email body to 200 words. "
                           f"Sign of with your name; Oksana"
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
# Step 3: Create Google calendar event
# --------------------------------------------------------------


class GoogleCalendarIntegrator:
    """A simplified Google Calendar integration handler."""

    SCOPES = ['https://www.googleapis.com/auth/calendar.events']
    TOKEN_FILE = 'token.pickle'
    CREDENTIALS_FILE = 'credentials.json'

    def __init__(self):
        self.service = self._get_calendar_service()

    def _get_calendar_service(self):
        """Initialize and return the Google Calendar service with proper authentication."""
        creds = None

        if os.path.exists(self.TOKEN_FILE):
            with open(self.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.CREDENTIALS_FILE, self.SCOPES)
                creds = flow.run_local_server(port=0)

            with open(self.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)

        return build('calendar', 'v3', credentials=creds)

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

    try:
        calendar_integrator = GoogleCalendarIntegrator()

        def parse_datetime(date_str, time_str):
            """Convert various date & time formats to ISO 8601 format."""
            date_formats = ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
            time_formats = ["%I:%M %p", "%I %p", "%H:%M"]

            # Try parsing date
            date_obj = next(
                (datetime.strptime(date_str, fmt) for fmt in date_formats if is_valid_format(date_str, fmt)), None)
            if not date_obj:
                raise ValueError(f"Unknown date format: {date_str}")

            # Try parsing time
            time_obj = next(
                (datetime.strptime(time_str, fmt).time() for fmt in time_formats if is_valid_format(time_str, fmt)),
                None)
            if not time_obj:
                raise ValueError(f"Unknown time format: {time_str}")

            return datetime.combine(date_obj.date(), time_obj).replace(tzinfo=timezone.utc).isoformat()

        def is_valid_format(date_or_time, fmt):
            """Check if a string matches a given date/time format."""
            try:
                datetime.strptime(date_or_time, fmt)
                return True
            except ValueError:
                return False



        full_iso_date = parse_datetime(event_details.date, event_details.time)
        calendar_event = calendar_integrator.service.events().insert(
            calendarId='primary',
            body={
                'summary': event_details.name,
                "description": event_info.description,
                'start': {
                    'dateTime': full_iso_date,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': full_iso_date,
                    'timeZone': 'UTC',
                },
                'location': getattr(event_details, 'location', ''),
                #'attendees': [{'name': person} for person in event_details.attendees],
                'reminders': {'useDefault': True}
            },
            sendUpdates='all'
        ).execute()
        logger.info(f"Calendar event created successfully with ID: {calendar_event.get('id')}")
        event_details.calendar_link = calendar_event.get('htmlLink')



    except Exception as e:
        logger.error(f"Failed to create calendar event: {str(e)}")
        event_details.calendar_link = None
        sys.exit()


    # Step 3: Generate confirmation email
    confirmation_email = generate_confirmation_email(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation_email

# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

#user_input = "How about scheduling a 90-minute meeting this Friday at 10 AM with Alex and Sarah to review the latest product updates?"
#user_input = "Can you send an email to Jane to prepare to discuss the project roadmap?"
user_input = input("Enter your input: ")

result = process_event(user_input)
if result:
    for email in result:
        print(f"Email to {email.invitee}")
        print(f"Subject: {email.subject}")
        print(f"Body: {email.body}")
        # if email.calendar_link:
        #     print(f"Calendar link: {email.calendar_link}")
        print("\n")
else:
    print("This doesn't appear to be a calendar event request.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

#user_input = "Can you send an email to Jane to prepare to discuss the project roadmap?"



###TODO:
# - figure out atendees/participants: if noone - no email, just calendar event
# - add location to the event details
# - if there are atendees - invent emails?
