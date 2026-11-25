import json
import os
import base64
import time
import openai


# --- Configuration ---
# Ensure you set your OpenAI API key before running the script
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"  # Replace with your OpenRouter API URL

# Use a model that supports image input
MODEL_NAME = "openai/gpt-5" 
INPUT_FILE = "benchmark.json"
OUTPUT_FILE = "gpt5_output.jsonl"
IMAGE_FOLDER = "./img" # Images are in the same directory as the script

NUM_SECONDS_TO_SLEEP = 0.5

# --- Prompt Templates ---
CONVERSATION_PROMPT = """
You are an expert neurologist. Your task is to analyze the provided EEG image and answer the user's question.

Follow these rules strictly:
1.  **Be Direct and Concise:** Answer the question directly. Limit your response to 1-3 sentences.
2.  **State the Conclusion First:** If the question is a yes/no or a classification question, state your conclusion at the beginning (e.g., "No, this is not an ictal pattern because...").
3.  **Focus on Evidence:** Base your answer only on the visual evidence in the EEG segment.
4.  **No Boilerplate:** Do not introduce yourself or explain what an EEG is. Do not list all theoretical possibilities.
5.  **Assume Expertise:** Assume you are communicating with another medical professional. Use appropriate terminology but avoid unnecessary jargon.

Answer the following question based on the EEG image provided:
{text}

Your Response:
"""

SELECTION_PROMPT = """
You are an expert neurologist. Based on the provided 10-second EEG image and the following multiple-choice question, state the correct option and you can provide a brief, one-sentence reason, cross-referencing the image.

Question:
{text}

Your Response (e.g., "**C.** Justification based on the image..."):
"""

SUMMARY_PROMPT = """
You are an expert neurologist. Your task is to provide a concise summary of the key findings from the provided EEG segment.

Follow these rules:
1.  **Be Objective:** Describe the visual data presented in the EEG.
2.  **Structure:** Start by commenting on the data quality (e.g., good, poor, limited by artifact). Then, describe any notable background rhythms, transients, or patterns observed.
3.  **Conciseness:** Keep the summary to 2-4 sentences.
4.  **No Interpretation without Evidence:** Only report what is visually present. If the recording is obscured, state that the interpretation is limited.
5.  **Assume Expertise:** Use standard EEG terminology.

This an example of a good summary:
"The channel layout uses the standard 10-20 system with an average reference. The overall data quality is good, with some focal low-voltage activity noted in the central midline region between 2.8 and 6.0 seconds and minor high-frequency muscle artifact over the frontopolar leads. The background activity is stable and predominantly composed of low-amplitude beta activity at approximately 14.6 Hz. A persistent interhemispheric asymmetry is present, characterized by consistently higher amplitudes over the left frontal and temporal regions compared to their homologous right-sided counterparts, as well as slower frequencies over the left temporal region. No definite epileptiform discharges or physiological sleep waves are identified in this segment."

Provide a summary for the EEG image provided.
{text} 
Your Summary:
"""

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def encode_image(image_path: str) -> str:
    """Encodes the image file to a base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_openai_multimodal_response(prompt_template: str, text: str, base64_image: str) -> str:
    """Calls the OpenAI multimodal API and returns the generated text."""
    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_template.format(text=text)
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_tokens=5000
            )
            break
        except openai.RateLimitError:
            print("Rate limit reached. Sleeping for 5 seconds.")
            time.sleep(5)
            pass
        except Exception as e:
            print(e)
            time.sleep(NUM_SECONDS_TO_SLEEP)

    return response.choices[0].message.content


def process_benchmark_with_images():
    """Reads the benchmark file and corresponding images, processes each entry, and writes to a jsonl file."""
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            benchmark_data = json.load(f_in)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{INPUT_FILE}'.")
        return

    # if os.path.isfile(OUTPUT_FILE):
    #     cur_ouput = [json.loads(line) for line in open(OUTPUT_FILE, encoding='utf-8')]
    # else :
    #     cur_ouput = []

    output_file = open(OUTPUT_FILE, 'a', encoding='utf-8')
    for image_key, data in benchmark_data.items():
        print(f"Processing: {image_key}")

        # Build image path and check if the file exists
        image_filename = image_key.replace('.npy', '.jpg')
        image_path = os.path.join(IMAGE_FOLDER, image_filename)

        if not os.path.exists(image_path):
            print(f"  - Warning: Image file not found: {image_path}. Skipping this entry.")
            continue

        # Encode the image
        base64_image = encode_image(image_path)

        # Prepare input text
        conversation_text = "\n".join(data['conversation'][0].replace('**User:**', '').strip())
        selection_text = "\n".join(data['selection'][0].strip())
        summary_text = ''

        # Call the API for each field
        print("  - Generating conversation...")
        conv_output = get_openai_multimodal_response(CONVERSATION_PROMPT, conversation_text, base64_image)
        
        print("  - Generating selection...")
        sel_output = get_openai_multimodal_response(SELECTION_PROMPT, selection_text, base64_image)
        
        print("  - Generating summary...")
        sum_output = get_openai_multimodal_response(SUMMARY_PROMPT, summary_text, base64_image)

        # Build the output object
        output_record = {
            "image": image_key,
            "output": {
                "conversation": conv_output,
                "selection": sel_output,
                "summary": sum_output
            }
        }

        # Write the result to the jsonl file
        output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
        output_file.flush()

        print(f"  - Finished and wrote to {OUTPUT_FILE}: {image_key}")

if __name__ == "__main__":
    process_benchmark_with_images()