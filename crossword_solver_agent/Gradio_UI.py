#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import mimetypes
import os
import re
import shutil
from typing import Optional

import PIL

from smolagents.agent_types import AgentAudio, AgentImage, AgentText, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={"title": "📝 Execution Logs", "parent_id": parent_id, "status": "done"},
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={"title": "💥 Error", "parent_id": parent_id, "status": "done"},
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count") and agent.model.last_input_token_count is not None:
            total_input_tokens += agent.model.last_input_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
        
        if hasattr(agent.model, "last_output_token_count") and agent.model.last_output_token_count is not None:
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=f"**Final answer:** {str(final_answer)}")


class GradioUI:
    """A Gradio interface for the crossword solver agent"""

    def __init__(self, agent: MultiStepAgent):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent

    def interact_with_agent(self, image, messages):
        """Handle interaction with the crossword solver agent"""
        import gradio as gr
        import tempfile
        import os

        if image is None:
            messages.append(gr.ChatMessage(role="assistant", content="Please upload a crossword puzzle image."))
            yield messages
            return

        # Debug: Show image info
        messages.append(gr.ChatMessage(
            role="assistant", 
            content=f"📄 DEBUG: Received image of type {type(image)} with size {image.size if hasattr(image, 'size') else 'unknown'}"
        ))
        yield messages

        # Create the task prompt with the image
        task = "Solve this New Yorker crossword puzzle."
        
        try:
            # Debug: Show task start
            messages.append(gr.ChatMessage(
                role="assistant",
                content="🔍 DEBUG: Starting crossword solving process..."
            ))
            yield messages

            # Save the PIL image to a temporary file that Gradio can use
            temp_dir = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_dir, "crossword.png")
            image.save(temp_image_path)
            
            # Add the user message with text only
            messages.append(gr.ChatMessage(role="user", content=task))
            yield messages
            
            # Add image as a separate system message
            messages.append(gr.ChatMessage(
                role="user", 
                content=(temp_image_path,)  # Tuple with image path
            ))
            yield messages

            # Debug: Show agent call
            messages.append(gr.ChatMessage(
                role="assistant",
                content="⚙️ DEBUG: Calling agent with image..."
            ))
            yield messages

            # Stream the agent's responses
            for msg in stream_to_gradio(
                self.agent,
                task=task,
                reset_agent_memory=True,
                additional_args={"crossword_image": image}
            ):
                messages.append(msg)
                yield messages

        except Exception as e:
            # Debug: Show any errors
            messages.append(gr.ChatMessage(
                role="assistant",
                content=f"❌ ERROR: An error occurred:\n```\n{str(e)}\n```"
            ))
            yield messages
            
            # Debug: Show error details
            import traceback
            messages.append(gr.ChatMessage(
                role="assistant",
                content=f"📋 Stack trace:\n```\n{traceback.format_exc()}\n```"
            ))
            yield messages

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        import gradio as gr

        with gr.Blocks(title="Crossword Solver", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🧩 New Yorker Crossword Solver
            
            Upload an image of a New Yorker style crossword puzzle, and I'll help you solve it!
            
            1. The image should contain both the puzzle grid and the clues
            2. Make sure the text is clear and readable
            3. Wait for the solution to be generated
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    # Image input for the crossword puzzle
                    image_input = gr.Image(
                        label="Upload Crossword Puzzle",
                        type="pil",
                        sources="upload"  # Only allow file upload
                    )

                    # Debug output
                    debug_output = gr.Markdown("Debug information will appear in the chat below")

            # Chat interface to show the solving process
            stored_messages = gr.State([])
            chatbot = gr.Chatbot(
                label="Solving Process",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/agents-course/course-images/resolve/main/en/communication/Alfred.png"
                ),
                height=600,
                show_copy_button=True,
                render_markdown=True
            )

            # Submit button
            submit_btn = gr.Button("Solve Crossword", variant="primary")
            
            # Handle submission
            submit_btn.click(
                self.interact_with_agent,
                [image_input, chatbot],
                [chatbot]
            )

            # Clear button
            clear_btn = gr.Button("Clear")
            clear_btn.click(
                lambda: (None, []),
                outputs=[image_input, chatbot]
            )

        demo.launch(debug=True, share=True, **kwargs)


__all__ = ["stream_to_gradio", "GradioUI"]