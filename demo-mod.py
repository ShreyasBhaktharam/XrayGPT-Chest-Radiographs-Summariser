import argparse
import os
import random
import json
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from xraygpt.common.config import Config
from xraygpt.common.dist_utils import get_rank
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from xraygpt.datasets.builders import *
from xraygpt.models import *
from xraygpt.processors import *
from xraygpt.runners import *
from xraygpt.tasks import *


def parse_args():
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


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# Function to save feedback to file for future training
def save_feedback_to_file(feedback_entry):
    """Save feedback to a file for future training"""
    os.makedirs("feedback_data", exist_ok=True)
    filename = f"feedback_data/feedback_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(feedback_entry, f, indent=2)


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

print(f"Setting default CUDA device to: cuda:{args.gpu_id}")
torch.cuda.set_device(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.openi.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list, feedback_state):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def upload_img(gr_img, text_input, chat_state, feedback_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def gradio_ask(user_message, chatbot, chat_state, feedback_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    if chat_state is None:
        return gr.update(interactive=True, placeholder='Please upload an image first!'), chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature, feedback_state):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    
    # Show feedback buttons after model generates response
    return chatbot, chat_state, img_list, feedback_state, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def accept_response(chatbot, chat_state, feedback_state):
    if len(chatbot) > 0:
        # Get the last query and response
        last_query = chatbot[-1][0]
        last_response = chatbot[-1][1]
        
        # Add feedback entry
        feedback_entry = {
            "query": last_query,
            "response": last_response,
            "feedback_type": "accept",
            "timestamp": time.time()
        }
        
        # Update feedback state
        feedback_state = feedback_state + [feedback_entry]
        
        # Save to file in real-time
        save_feedback_to_file(feedback_entry)
    
    return chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

def modify_response(chatbot, chat_state, modified_text, feedback_state):
    if len(chatbot) > 0:
        original_response = chatbot[-1][1]
        last_query = chatbot[-1][0]
        
        # Update the chatbot with modified text
        chatbot[-1][1] = modified_text
        
        # Add feedback entry
        feedback_entry = {
            "query": last_query,
            "original_response": original_response,
            "modified_response": modified_text,
            "feedback_type": "modify",
            "timestamp": time.time()
        }
        
        # Update feedback state
        feedback_state = feedback_state + [feedback_entry]
        
        # Save to file in real-time
        save_feedback_to_file(feedback_entry)
    
    return chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

def reject_response(chatbot, chat_state, replacement_text, feedback_state):
    if len(chatbot) > 0:
        original_response = chatbot[-1][1]
        last_query = chatbot[-1][0]
        
        # Update the chatbot with replacement text
        chatbot[-1][1] = replacement_text if replacement_text else "Response rejected."
        
        # Add feedback entry
        feedback_entry = {
            "query": last_query,
            "original_response": original_response,
            "replacement_response": replacement_text if replacement_text else "",
            "feedback_type": "reject",
            "timestamp": time.time()
        }
        
        # Update feedback state
        feedback_state = feedback_state + [feedback_entry]
        
        # Save to file in real-time
        save_feedback_to_file(feedback_entry)
    
    return chatbot, chat_state, feedback_state, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(interactive=True)

def set_example_xray(example: list) -> dict:
    return gr.Image.update(value=example[0])

def set_example_text_input(example_text: str) -> dict:
    return gr.Textbox.update(value=example_text[0])

title = """<h1 align="center">Demo of XrayGPT</h1>"""
description = """<h3>Upload your X-Ray images and start asking queries!</h3>"""
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

    # Add feedback state
    feedback_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil")
            upload_button = gr.Button(value="Upload and Ask Queries", interactive=True, variant="primary")
            clear = gr.Button("Reset")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='XrayGPT')
            text_input = gr.Textbox(label='User', placeholder='Please upload your X-Ray image.', interactive=False)
            
            # Feedback Components
            with gr.Row(visible=False) as feedback_row:
                accept_btn = gr.Button("Accept", variant="success")
                modify_btn = gr.Button("Modify", variant="secondary")
                reject_btn = gr.Button("Reject", variant="stop")
            
            modified_text = gr.Textbox(label="Edit Response", visible=False)
            replacement_text = gr.Textbox(label="Write New Response", visible=False)

    with gr.Row():
        example_xrays = gr.Dataset(components=[image], label="X-Ray Examples",
                                    samples=[
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img1.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img2.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img3.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img4.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img5.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img6.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img7.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img8.png")],
                                        [os.path.join(os.path.dirname(__file__), "images/example_test_images/img9.png")],
                                    ])
        
    with gr.Row():
        example_texts = gr.Dataset(components=[gr.Textbox(visible=False)],
                                    label="Prompt Examples",
                                    samples=[
                                        ["Describe the given chest x-ray image in detail."],
                                        ["Take a look at this chest x-ray and describe the findings and impression."],
                                        ["Could you provide a detailed description of the given x-ray image?"],
                                        ["Describe the given chest x-ray image as detailed as possible."],
                                        ["What are the key findings in this chest x-ray image?"],
                                        ["Could you highlight any abnormalities or concerns in this chest x-ray image?"],
                                        ["What specific features of the lungs and heart are visible in this chest x-ray image?"],
                                        ["What is the most prominent feature visible in this chest x-ray image, and how is it indicative of the patient's health?"],
                                        ["Based on the findings in this chest x-ray image, what is the overall impression?"],
                                    ],)
    
    # Handle Feedback Button Events
    accept_btn.click(
        accept_response, 
        [chatbot, chat_state, feedback_state], 
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )
    
    modify_btn.click(
        lambda chatbot: chatbot[-1][1] if len(chatbot) > 0 else "",
        [chatbot],
        [modified_text]
    ).then(
        lambda: (gr.update(visible=True), gr.update(visible=False)), 
        None, 
        [modified_text, replacement_text]
    )
    
    modified_text.submit(
        modify_response,
        [chatbot, chat_state, modified_text, feedback_state],
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )
    
    reject_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        None,
        [modified_text, replacement_text]
    )
    
    replacement_text.submit(
        reject_response,
        [chatbot, chat_state, replacement_text, feedback_state],
        [chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text, text_input]
    )
    
    # Original event handlers updated with feedback components
    example_xrays.click(fn=set_example_xray, inputs=example_xrays, outputs=example_xrays.components)

    upload_button.click(
        upload_img, 
        [image, text_input, chat_state, feedback_state], 
        [image, text_input, upload_button, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text]
    )
    
    example_texts.click(
        set_example_text_input, 
        inputs=example_texts, 
        outputs=text_input
    ).then(
        gradio_ask, 
        [text_input, chatbot, chat_state, feedback_state], 
        [text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text]
    ).then(
        gradio_answer, 
        [chatbot, chat_state, img_list, num_beams, temperature, feedback_state], 
        [chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text]
    )
    
    text_input.submit(
        gradio_ask, 
        [text_input, chatbot, chat_state, feedback_state], 
        [text_input, chatbot, chat_state, feedback_state, feedback_row, modified_text, replacement_text]
    ).then(
        gradio_answer, 
        [chatbot, chat_state, img_list, num_beams, temperature, feedback_state], 
        [chatbot, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text]
    )
    
    clear.click(
        gradio_reset, 
        [chat_state, img_list, feedback_state], 
        [chatbot, image, text_input, upload_button, chat_state, img_list, feedback_state, feedback_row, modified_text, replacement_text], 
        queue=False
    )
    
    gr.Markdown(disclaimer)

demo.launch(share=True, enable_queue=True)
