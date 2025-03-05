# Calendar Agent

A simple Python-only calendar assistant that processes natural language requests to schedule meetings and events using OpenAI/Groq LLMs and Google Calendar integration.
Main purpose was to write an agent without using any LLM frameworks, just pure Python.

## Prerequisites

- Python 3.8+
- Google Calendar API credentials
- Groq API key

## Setup

1. Install required dependencies:
```bash
pip install google-auth-httplib2 google-auth-oauthlib google-api-python-client pydantic openai groq python-dotenv
``` 

2. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

3. Set up Google Calendar API:
   - Create a project in Google Cloud Console
   - Enable the Google Calendar API
   - Create credentials (OAuth 2.0 Client ID)
   - Download the credentials and save as `credentials.json` in the project directory

## Usage

The agent can be used by running the script and providing a natural language input:
```bash
python calendar_agent_prompt_chaining.py
```

Example inputs:

"Let's schedule a 2h meeting next Tuesday at 2pm with Jane and John to discuss the project roadmap."
"Schedule a team sync for tomorrow at 3pm with the development team."

## How It Works

The agent processes requests in three main stages:

1. **Event Extraction**: Analyzes if the input text describes a calendar event
2. **Event Details Parsing**: Extracts specific details like date, time, attendees
3. **Calendar Integration**: Creates the event in Google Calendar
4. **Email Generation**: Creates confirmation emails for all attendees


## Limitations

- Currently supports single timezone (UTC)
- Requires manual Google Calendar authentication on first run
- Needs valid Google Calendar API credentials
- Requires internet connection for API calls

TODO:
 - figure out atendees/participants: if noone - no email, just calendar event
 - add location to the event details
 - if there are atendees - invent email addreses?
