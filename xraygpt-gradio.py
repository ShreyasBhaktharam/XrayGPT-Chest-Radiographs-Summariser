import argparse
import os
import random
import json
import time
import base64 # Added
import io     # Added
import requests # Added
from PIL import Image # Make sure PIL is available for image operations

import numpy as np
import torch # Still needed for torch.backends.cudnn if setup_seeds is kept
import torch.backends.cudnn as cudnn
import gradio as gr

# Assuming these are still needed for CONV_VISION or other non-model utilities
from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry # May not be needed if all registry calls are removed
from xraygpt.conversation.conversation import Chat, CONV_VISION # CONV_VISION is key

# These might not be needed anymore if model loading is removed, review carefully
# from xraygpt.datasets.builders import *
# from xraygpt.models import *
# from xraygpt.processors import *
# from xraygpt.runners import *
# from xraygpt.tasks import *

# Define API endpoint for your FastAPI server
API_ENDPOINT = "http://localhost:8800" # Ensure this matches your FastAPI server address

def parse_args_original(): # Renamed to avoid conflict if not used
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def setup_seeds_original(config): # Renamed, may not be fully applicable
    # This function used config.run_cfg.seed. If 'config' is removed, this needs adjustment or removal.
    # For now, let's assume a default seed if config is not available.
    seed = getattr(config, 'run_cfg', {}).get('seed', 42) + get_rank() # Basic fallback
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

# Function to save feedback to file for future training (remains the same)
def save_feedback_to_file(feedback_entry):
    """Save feedback to a file for future training"""
    os.makedirs("feedback_data", exist_ok=True)
    filename = f"feedback_data/feedback_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(feedback_entry, f, indent=2)

# ========================================
#       Model Initialization (REMOVED)
# ========================================
# The local model initialization (args, cfg, model, vis_processor, chat) is removed.
# The Gradio app will now act as a client to the FastAPI server.
print("Gradio App running in Client Mode: Model is served by FastAPI.")


# ========================================
#           Gradio Functions
# ========================================
# ... (other imports and function definitions) ...
'''
def set_example_text_input(example_data):
    """
    Takes the data from the clicked example in the gr.Dataset 
    and returns an update for the text_input Textbox.
    The 'example_data' from a gr.Dataset with a single Textbox component
    will typically be a list containing the string, e.g., ["Example prompt text"].
    """
    if isinstance(example_data, list) and len(example_data) > 0:
        # Assuming the first element of the list is the text string
        return gr.Textbox.update(value=example_data[0])
    elif isinstance(example_data, str): # If for some reason it's passed as a direct string
        return gr.Textbox.update(value=example_data)
    return gr.Textbox.update() # Default to no change or clear if data is unexpected

# ... (other imports and function definitions like save_feedback_to_file, gradio_reset, etc.) ...
'''
def set_example_xray(example_data_list: list):
    """
    Takes the data from the clicked example in the gr.Dataset (example_xrays)
    and returns an update for the main image component.
    The 'example_data_list' from a gr.Dataset with a single gr.Image component
    will typically be a list containing the image path or PIL image, 
    e.g., ["path/to/example/image.png"].
    """
    if isinstance(example_data_list, list) and len(example_data_list) > 0:
        # The value from the dataset for an image component is usually the filepath
        return gr.Image.update(value=example_data_list[0])
    return gr.Image.update() # Default to no change or clear if data is unexpected

def set_example_text_input(example_data_list: list): # Ensure this is also present
    """
    Takes the data from the clicked example in the gr.Dataset (example_texts)
    and returns an update for the text_input Textbox.
    """
    if isinstance(example_data_list, list) and len(example_data_list) > 0:
        return gr.Textbox.update(value=example_data_list[0])
    return gr.Textbox.update()

def gradio_reset(chat_state_val, img_list_val, feedback_state_val):
    # chat_state_val is the CONV_VISION object
    if chat_state_val is not None:
        chat_state_val.messages = [] # Clear messages in the conversation object
    
    # img_list_val is the list holding the base64 image
    new_img_list = [] # Reset to an empty list

    # The outputs for clear.click are:
    # [chatbot, image, text_input, upload_button, chat_state, img_list, feedback_state, 
    #  feedback_row, modified_text, replacement_text] (10 items)
    return None, \
           gr.update(value=None, interactive=True), \
           gr.update(placeholder='Please upload your image first', interactive=False), \
           gr.update(value="Upload & Start Chat", interactive=True), \
           chat_state_val, \
           new_img_list, \
           feedback_state_val, \
           gr.update(visible=False), \
           gr.update(visible=False), \
           gr.update(visible=False)

def upload_img(gr_pil_image, text_input_val, current_chat_state, current_feedback_state):
    # The outputs for upload_button.click are:
    # [image, text_input, upload_button, chat_state, img_list, feedback_state, 
    #  feedback_row, modified_text, replacement_text] (9 items)
    if gr_pil_image is None:
        return gr.update(value=None), \
               gr.update(interactive=True), \
               gr.update(interactive=True), \
               None, \
               None, \
               current_feedback_state, \
               gr.update(visible=False), \
               gr.update(visible=False), \
               gr.update(visible=False)

    # Convert PIL image to base64 string
    try:
        buffered = io.BytesIO()
        gr_pil_image.save(buffered, format="PNG") # Or JPEG, ensure API handles it
        base64_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {e}")
        # Handle error appropriately, maybe update UI to show error
        # For now, let's prevent further processing
        return gr.update(value=gr_pil_image), \
               gr.update(interactive=True, placeholder=f"Error encoding image: {e}"), \
               gr.update(interactive=True), \
               None, \
               None, \
               current_feedback_state, \
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    new_chat_state = CONV_VISION.copy() # Initialize a new conversation state for Gradio
    new_img_list = [base64_img_str]     # Store the base64 image in a list

    print("Image uploaded and encoded to base64.")
    
    return gr.update(interactive=False), \
           gr.update(interactive=True, placeholder='Type and press Enter'), \
           gr.update(value="Start Chatting", interactive=False), \
           new_chat_state, \
           new_img_list, \
           current_feedback_state, \
           gr.update(visible=False), \
           gr.update(visible=False), \
           gr.update(visible=False)

def gradio_ask(user_message, chatbot_display_history, current_chat_state, current_feedback_state):
    # Outputs for text_input.submit -> gradio_ask are:
    # [text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text] (7 items)

    if not user_message.strip():
        return gr.update(interactive=True, placeholder='Input should not be empty!'), \
               chatbot_display_history, current_chat_state, current_feedback_state, \
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    if current_chat_state is None:
        # This indicates image was not uploaded first, or state was lost
        return gr.update(interactive=True, placeholder='Please upload an image first!'), \
               chatbot_display_history, current_chat_state, current_feedback_state, \
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    # Initialize chatbot_display_history if it's None (e.g., first turn)
    new_chatbot_display_history = list(chatbot_display_history) if chatbot_display_history is not None else []

    # Append user message to the CONV_VISION object (chat_state)
    # This object tracks the full conversation for context if needed by API, 
    # or just to get the latest message.
    current_chat_state.append_message(current_chat_state.roles[0], user_message) # roles[0] is typically 'USER'
    
    # Update the Gradio chatbot display list
    new_chatbot_display_history.append([user_message, None]) # Bot response is None for now
    
    print(f"gradio_ask: Appended user message to chat_state. Current messages: {current_chat_state.messages}")
    print(f"gradio_ask: Updated chatbot display history for UI: {new_chatbot_display_history}")

    return '', new_chatbot_display_history, current_chat_state, current_feedback_state, \
           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def gradio_answer(chatbot_display_history, current_chat_state, current_img_list, num_beams, temperature, current_feedback_state):
    # Outputs for .then(gradio_answer) are:
    # [chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text] (7 items)

    current_chatbot_display_val = list(chatbot_display_history) if chatbot_display_history is not None else []
    
    # Prepare default outputs (important for returning consistently)
    outputs = [
        gr.update(value=current_chatbot_display_val), current_chat_state, current_img_list, current_feedback_state,
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    ]

    # Validations
    if not current_chat_state or not current_chat_state.messages:
        error_msg = "Error: Conversation context (chat_state) is missing or empty."
        print(error_msg)
        if current_chatbot_display_val and len(current_chatbot_display_val[-1]) == 2:
            current_chatbot_display_val[-1][1] = error_msg
        else:
            current_chatbot_display_val.append([None, error_msg]) # Add error as a new bot message
        outputs[0] = gr.update(value=current_chatbot_display_val)
        return tuple(outputs)

    if not current_img_list or not current_img_list[0]:
        error_msg = "Error: No image provided (img_list is empty or image string is missing)."
        print(error_msg)
        if current_chatbot_display_val and len(current_chatbot_display_val[-1]) == 2:
            current_chatbot_display_val[-1][1] = error_msg
        else:
            current_chatbot_display_val.append([None, error_msg])
        outputs[0] = gr.update(value=current_chatbot_display_val)
        return tuple(outputs)

    # Extract the last user message from current_chat_state to send to API
    # Assuming the last message in current_chat_state.messages is [role, content]
    last_message_role, last_message_content = current_chat_state.messages[-1]
    if last_message_role != current_chat_state.roles[0]: # If not USER, something is off
        print(f"Warning in API call: Last message in chat_state was from '{last_message_role}', "
              f"expected user role '{current_chat_state.roles[0]}'. Using its content anyway.")
    
    user_message_for_api = last_message_content
    base64_image_for_api = current_img_list[0]

    payload = {
        "image": base64_image_for_api,
        "message": user_message_for_api,
        "parameters": {
            "num_beams": num_beams,
            "temperature": temperature,
            "max_new_tokens": 300, # Default from your FastAPI
            "max_length": 2000     # Default from your FastAPI
        }
    }

    llm_api_response_text = "Error: Could not connect to API or API returned an error." # Default error

    try:
        print(f"Calling API: {API_ENDPOINT}/generate for message: '{user_message_for_api}'")
        api_start_time = time.time()
        response = requests.post(f"{API_ENDPOINT}/generate", json=payload, timeout=300) # 5-minute timeout
        api_duration = time.time() - api_start_time
        print(f"API call completed in {api_duration:.2f} seconds.")

        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        response_json = response.json()
        llm_api_response_text = response_json.get("response", "Error: 'response' key missing in API JSON.")
        
        # Update the CONV_VISION object (current_chat_state) with the bot's response
        current_chat_state.append_message(current_chat_state.roles[1], llm_api_response_text) # roles[1] is 'ASSISTANT'
        print(f"gradio_answer: Updated chat_state with bot response: {current_chat_state.messages}")

    except requests.exceptions.Timeout:
        error_msg = "Error: API request timed out."
        print(error_msg)
        llm_api_response_text = error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Error: API request failed: {e}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get("detail", e.response.text) # Try to get FastAPI's detail
                error_msg += f" (Detail: {error_detail})"
            except ValueError: # If response is not JSON
                error_msg += f" (Status: {e.response.status_code}, Response: {e.response.text})"
        print(error_msg)
        llm_api_response_text = error_msg
    except Exception as e:
        error_msg = f"An unexpected error occurred in gradio_answer: {e}"
        print(error_msg)
        llm_api_response_text = error_msg

    # Update the chatbot display list for the UI
    if current_chatbot_display_val and len(current_chatbot_display_val[-1]) == 2:
        current_chatbot_display_val[-1][1] = llm_api_response_text
    else:
        # This might happen if gradio_ask didn't properly add [user, None]
        # or if called without a preceding user message in the display
        current_chatbot_display_val.append([user_message_for_api, llm_api_response_text])
        print("Warning: Appended directly to chatbot_display_val in gradio_answer as last turn was not [user, None].")

    print(f"gradio_answer: Final chatbot display value for UI: {current_chatbot_display_val}")
    
    outputs[0] = gr.update(value=current_chatbot_display_val) # Update chatbot display
    outputs[1] = current_chat_state # Pass back updated chat_state (CONV_VISION obj)
    # Other outputs (img_list, feedback_state, visibility updates) remain as set initially
    
    return tuple(outputs)


# --- Feedback Functions (largely unchanged, but ensure they use chatbot_display_history) ---
# These functions use the chatbot display history. If you want them to interact
# with an API for feedback, that would be an additional modification.
# For now, they save feedback locally.

def accept_response(chatbot_display_history, current_chat_state, current_feedback_state):
    if not chatbot_display_history or len(chatbot_display_history) == 0:
        # Return 7 items (chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input)
        return chatbot_display_history, current_chat_state, current_feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

    last_query = chatbot_display_history[-1][0]
    last_response = chatbot_display_history[-1][1]
    
    feedback_entry = {
        "query": last_query, "response": last_response,
        "feedback_type": "accept", "timestamp": time.time()
    }
    new_feedback_state = (current_feedback_state if current_feedback_state is not None else []) + [feedback_entry]
    save_feedback_to_file(feedback_entry)
    
    return chatbot_display_history, current_chat_state, new_feedback_state, \
           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
           gr.update(interactive=True) # text_input

def modify_response(chatbot_display_history, current_chat_state, modified_text, current_feedback_state):
    if not chatbot_display_history or len(chatbot_display_history) == 0:
        return chatbot_display_history, current_chat_state, current_feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

    original_response = chatbot_display_history[-1][1]
    last_query = chatbot_display_history[-1][0]
    
    chatbot_display_history[-1][1] = modified_text # Update the UI
    
    feedback_entry = {
        "query": last_query, "original_response": original_response,
        "modified_response": modified_text, "feedback_type": "modify",
        "timestamp": time.time()
    }
    new_feedback_state = (current_feedback_state if current_feedback_state is not None else []) + [feedback_entry]
    save_feedback_to_file(feedback_entry)
    
    return chatbot_display_history, current_chat_state, new_feedback_state, \
           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
           gr.update(interactive=True) # text_input

def reject_response(chatbot_display_history, current_chat_state, replacement_text, current_feedback_state):
    if not chatbot_display_history or len(chatbot_display_history) == 0:
        return chatbot_display_history, current_chat_state, current_feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

    original_response = chatbot_display_history[-1][1]
    last_query = chatbot_display_history[-1][0]
    
    final_text = replacement_text if replacement_text else "Response rejected."
    chatbot_display_history[-1][1] = final_text # Update the UI
    
    feedback_entry = {
        "query": last_query, "original_response": original_response,
        "replacement_response": replacement_text if replacement_text else "",
        "feedback_type": "reject", "timestamp": time.time()
    }
    new_feedback_state = (current_feedback_state if current_feedback_state is not None else []) + [feedback_entry]
    save_feedback_to_file(feedback_entry)
    
    return chatbot_display_history, current_chat_state, new_feedback_state, \
           gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
           gr.update(interactive=True) # text_input

# --- UI Layout (largely unchanged) ---
title = """<h1 align="center">Demo of XrayGPT (API Client)</h1>""" # Updated title
description = """<h3>Upload your X-Ray images and start asking queries! Responses are generated by a remote API.</h3>"""
disclaimer = """ 
            <h1 >Terms of Use:</h1>
            <ul> 
                <li>You acknowledge that the XrayGPT service is designed for research purposes with the ultimate aim of assisting medical professionals in their diagnostic process. It is important to note that the Service does not replace professional medical advice or diagnosis.</li>
                <li>XrayGPT utilizes advanced artificial intelligence algorithms (LLVM's) to carefully analyze and summarize X-ray images for medical diagnostic purposes. The results provided by the Service are derived from the thorough analysis conducted by the AI system, based on the X-ray images provided by the user.</li>
                <li>We strive to provide accurate and helpful results through XrayGPT. However, it is important to understand that we do not make any explicit warranties or representations regarding the effectiveness, reliability, or completeness of the results provided. Our aim is to continually improve and refine the Service to provide the best possible assistance to medical professionals.</li>
            </ul>
            <hr> 
            <h3 align="center">Designed and Developed by IVAL Lab, MBZUAI</h3>
            """

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    # States for conversation context, image list (base64 string), and feedback
    chat_state = gr.State() # Will hold the CONV_VISION object
    img_list = gr.State()   # Will hold a list with [base64_image_string]
    feedback_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil", label="Upload X-Ray Image") # type="pil" is good
            upload_button = gr.Button(value="Upload Image & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Reset Conversation")
            
            num_beams = gr.Slider(
                minimum=1, maximum=10, value=1, step=1,
                interactive=True, label="Beam Search Nums (for API)",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                interactive=True, label="Temperature (for API)",
            )

        with gr.Column():
            chatbot = gr.Chatbot(label='XrayGPT Conversations') # UI display component
            text_input = gr.Textbox(label='Your Query', placeholder='Please upload an image first.', interactive=False)
            
            with gr.Row(visible=False) as feedback_row:
                accept_btn = gr.Button("Accept", variant="success")
                modify_btn = gr.Button("Modify", variant="secondary")
                reject_btn = gr.Button("Reject", variant="stop")
            
            modified_text = gr.Textbox(label="Edit Response", visible=False, lines=3)
            replacement_text = gr.Textbox(label="Provide Correct Response", visible=False, lines=3)

    # Examples (remain the same, but their utility might change if model interaction is purely via API)
    # For now, keeping them as they are.
    with gr.Row():
        example_xrays = gr.Dataset(components=[image], label="X-Ray Examples",
                                    samples=[
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img1.png")],
                                        # ... (other image paths) ...
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img9.png")],
                                    ])
    with gr.Row():
        example_texts = gr.Dataset(components=[gr.Textbox(visible=False)],
                                   label="Prompt Examples",
                                   samples=[
                                       ["Describe the given chest x-ray image in detail."],
                                       # ... (other example prompts) ...
                                       ["Based on the findings in this chest x-ray image, what is the overall impression?"],
                                   ])
    
    # --- Event Handlers ---
    
    # outputs for upload_button.click (9 items):
    # image, text_input, upload_button, chat_state (State), img_list (State), feedback_state (State),
    # feedback_row, modified_text, replacement_text
    upload_button.click(
        upload_img, 
        [image, text_input, chat_state, feedback_state], 
        [image, text_input, upload_button, chat_state, img_list, feedback_state, 
         feedback_row, modified_text, replacement_text]
    )

    # Chain for text input: ask (updates UI and chat_state) -> answer (calls API, updates UI and chat_state)
    # outputs for .then(gradio_ask) (7 items):
    # text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text
    # outputs for .then(gradio_answer) (7 items):
    # chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text
    
    submit_event_chain = text_input.submit(
        gradio_ask, 
        [text_input, chatbot, chat_state, feedback_state], 
        [text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text]
    ).then(
        gradio_answer, 
        [chatbot, chat_state, img_list, num_beams, temperature, feedback_state], 
        [chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text],
        api_name="answer" # Keeps the Gradio Client API name if you use it
    )

    # Example texts should also trigger the same chain
    example_texts.click(
        fn=set_example_text_input, 
        inputs=[example_texts], 
        outputs=[text_input]
    ).then(
        gradio_ask, 
        [text_input, chatbot, chat_state, feedback_state], 
        [text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text]
    ).then(
        gradio_answer, 
        [chatbot, chat_state, img_list, num_beams, temperature, feedback_state], 
        [chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text]
    )
    
    # outputs for clear.click (10 items):
    # chatbot, image, text_input, upload_button, chat_state, img_list, feedback_state, 
    # feedback_row, modified_text, replacement_text
    clear.click(
        gradio_reset, 
        [chat_state, img_list, feedback_state], 
        [chatbot, image, text_input, upload_button, chat_state, img_list, feedback_state, 
         feedback_row, modified_text, replacement_text], 
        queue=False
    )

    # Feedback button event handlers
    # outputs for accept_btn.click (7 items):
    # chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input
    accept_btn.click(
        accept_response, 
        [chatbot, chat_state, feedback_state], 
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )
    
    modify_btn.click(
        lambda current_chatbot_val: current_chatbot_val[-1][1] if current_chatbot_val and len(current_chatbot_val) > 0 else "",
        [chatbot], # Input is the chatbot display history
        [modified_text] # Output is the textbox to populate
    ).then(
        lambda: (gr.update(visible=True), gr.update(visible=False)), 
        None, 
        [modified_text, replacement_text] # Show modified_text, hide replacement_text
    )
    
    modified_text.submit(
        modify_response,
        [chatbot, chat_state, modified_text, feedback_state],
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )
    
    reject_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)), # Hide modify, show replacement
        None,
        [modified_text, replacement_text]
    )
    
    replacement_text.submit(
        reject_response,
        [chatbot, chat_state, replacement_text, feedback_state],
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )

    example_xrays.click(fn=set_example_xray, inputs=[example_xrays], outputs=example_xrays.components)
        
    gr.Markdown(disclaimer)

# Setup and Launch (No local model args needed here for Gradio client)
if __name__ == "__main__":
    # Consider if setup_seeds is still relevant or how it should be handled
    # For now, commenting out as 'cfg' is not defined in this client-only version.
    # setup_seeds(cfg) # 'cfg' would not be defined here unless you load a client-side config

    print("Starting Gradio Client App for XrayGPT API...")
    demo.queue().launch(share=True, server_name="0.0.0.0") # enable_queue=True is default in launch() if queue() is called. server_name for wider access.
