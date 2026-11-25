import os
import torch
import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor


# ===== 1. Load Model =====
# Model loading only needs to be done once.
model_path = "CerebraGloss/CerebraGloss"
try:
    print("Loading model components...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ===== 2. Inference Function (Core Logic) =====
def run_inference(user_text, image_path=None, history=None):
    """
    Runs model inference and returns the response.

    Args:
        user_text (str): The user's input text.
        image_path (str, optional): Path to the image file.
        history (list, optional): Conversation history in the format [['user_msg', 'assistant_msg'], ...].

    Returns:
        str: The generated response from the model.
    """
    if history is None:
        history = []

    # Build Qwen message format
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for human, assistant in history:
        messages.append({"role": "user", "content": human})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})

    cur_msg_content = []
    if user_text:
        cur_msg_content.append({"type": "text", "text": user_text})
    if image_path:
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found at {image_path}. Skipping image.")
        else:
            pil_image = Image.open(image_path).convert("RGB")
            cur_msg_content.append({"type": "image", "image": pil_image})
    
    messages.append({"role": "user", "content": cur_msg_content})

    # Generate prompt
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare model input
    if image_path and os.path.exists(image_path):
        inputs = processor(text=[prompt_text], images=[pil_image], return_tensors="pt").to(model.device)
    else:
        inputs = processor(text=[prompt_text], return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 1024,
            do_sample=True,
            temperature=0.95,
            top_p=0.7,
        )

    # Decode and clean up the response
    response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response_text.strip()

# ===== 3. Command-Line Interface and Main Logic =====
def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL inference from the command line.")
    parser.add_argument("--text", "-t", type=str, help="The user's input text.")
    parser.add_argument("--image", "-i", type=str, help="The path to the image file.")
    args = parser.parse_args()

    # If no arguments are provided, enter interactive mode
    if not args.text and not args.image:
        print("Entering interactive mode. Type 'exit' to quit.")
        print("You can enter a command like 'What is this? | /path/to/image.jpg'")
        history = []
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break

            parts = user_input.split('|')
            input_text = parts[0].strip()
            image_path = parts[1].strip() if len(parts) > 1 else None

            if not input_text and not image_path:
                continue

            response = run_inference(input_text, image_path, history)
            print(f"Model: {response}")
            history.append([input_text, response])
    else:
        # Command-line mode
        if not args.text:
            print("Error: --text is required when running in command-line mode.")
            return

        print(f"Input Text: {args.text}")
        if args.image:
            print(f"Image Path: {args.image}")

        response = run_inference(args.text, args.image)
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    main()