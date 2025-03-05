# Chatbot and Meme Agent


A conversational AI agent that can engage in chat conversations and generate memes on demand. Built using the SmoLAgents framework and Gradio UI. Part of Hugging Face Agents course. Course is available at https://huggingface.co/learn/agents-course/

## Features

- Interactive chat interface powered by Qwen 2.5 Coder model
- Meme generation using Stable Diffusion
- Web search capabilities using DuckDuckGo
- Timezone conversion tool
- File upload support


## Prerequisites

- Python 3.8+
- Hugging Face API token
- DuckDuckGo API access


## Usage

1. Start the agent:
```bash
python app.py
```

2. Open your web browser and navigate to the provided Gradio interface URL

3. You can interact with the agent through:
- Text chat
- File uploads (supports PDF, DOCX, and TXT files)
- Meme generation requests
- Timezone queries

## Available Tools

### Meme Generator
Generate memes using natural language descriptions:
```python
meme = meme_generator("A funny cat wearing sunglasses")
```

### Flashcard Generator (not working)
Create study flashcards from Hugging Face course content:
```python
flashcards = generate_flashcards("https://huggingface.co/learn/course-name")
```

### Timezone Tool
Get current time in different timezones:
```python
time = get_current_time_in_timezone("America/New_York")
```

## Configuration

The agent's behavior can be customized through:

- `prompts.yaml`: Contains system prompts and planning templates
- `agent.json`: Configures tools, model parameters, and imports

## Model

The agent uses the Qwen 2.5 Coder model for natural language understanding and generation. Alternative models can be configured in `app.py`.

