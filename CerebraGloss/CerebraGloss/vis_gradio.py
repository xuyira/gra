import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer
from PIL import Image
import threading
import re
import shutil


# --- Force Gradio to use English ---
os.environ['GRADIO_LANGUAGE'] = 'en'

# ===== 1. Load Model =====
model_path = "CerebraGloss/CerebraGloss"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.eval()

# ===== 2. Chat Function (Streaming + Deduplication) =====
def stream_chat(user_text, image_path, history, temperature, top_p, max_length):
    if user_text is None:
        user_text = ""

    # Build Qwen message format
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    cur_msg_content = []
    if user_text:
        cur_msg_content.append({"type": "text", "text": user_text})
    if image_path is not None:
        pil_image = Image.open(image_path).convert("RGB")
        cur_msg_content.append({"type": "image", "image": pil_image})
    messages.append({"role": "user", "content": cur_msg_content})

    # Generate prompt
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # **Modified: Now the user message in history only contains text**
    user_display_message = user_text

    # Update Gradio history with the formatted user message
    chatbot_history = history + [[user_display_message, None]]

    # Prepare model input
    if image_path is not None:
        inputs = processor(text=[prompt_text], images=[pil_image], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[prompt_text], return_tensors="pt").to(model.device)

    # Stream generation
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    input_length = inputs["input_ids"].shape[1]
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_length=input_length + max_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Streaming output logic
    partial_output = ""
    for new_text in streamer:
        partial_output += new_text
        chatbot_history[-1][1] = partial_output
        yield chatbot_history, chatbot_history

# ---
# Function to automatically clean the static directory
# ---
def cleanup_static_directory():
    """Cleans up all files in the static directory before the app starts."""
    static_dir = "static"
    if os.path.exists(static_dir):
        # Iterate through all files and subdirectories
        for item in os.listdir(static_dir):
            item_path = os.path.join(static_dir, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # Delete file or link
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Delete subdirectory
            except Exception as e:
                print(f'Unable to delete {item_path}. Reason: {e}')
        print(f"Static directory '{static_dir}' has been cleaned.")
    else:
        print(f"Static directory '{static_dir}' does not exist, no cleanup needed.")

# ===== 3. Gradio UI =====
with gr.Blocks() as demo:
    # Removed the gr.Image component for the model logo.
    
    gr.Markdown("## CerebraGloss Demo")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                avatar_images=("figures/user_avatar.png", "figures/model_logo.png")
            )
            user_text = gr.Textbox(
                label="Enter Message", 
                placeholder="Please enter your message", 
                lines=2
            )

            image_uploader = gr.Image(
                label="Upload an Image (Optional)",
                type="filepath",
                image_mode="RGB",
                height=400,
            )
            
            submit_btn = gr.Button("Send")
            clear_btn = gr.Button("Clear Chat")
        with gr.Column(scale=1):
            temperature = gr.Slider(0.0, 2.0, value=0.95, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Top-p")
            max_length = gr.Slider(16, 2048, value=1024, step=16, label="Max Length")
            state = gr.State([])

    submit_btn.click(
        stream_chat,
        inputs=[user_text, image_uploader, state, temperature, top_p, max_length],
        outputs=[chatbot, state],
    )
    clear_btn.click(lambda: ([], [], None), None, [chatbot, state, image_uploader])
    
cleanup_static_directory()

demo.queue().launch(
    server_name="0.0.0.0", 
    server_port=7862, 
    share=True,
    allowed_paths=["./static"]
)