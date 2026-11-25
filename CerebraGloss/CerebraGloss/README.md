# CerebraGloss Demo

This is a demonstration project based on the CerebraGloss-3B model, providing two different running modes: a Gradio UI demo and a command-line demo for backend or script usage.

### Setup

1. **Install dependencies**: Creating a new environment is recommended.
   ```bash
   conda create -n cerebragloss python=3.10
   conda activate cerebragloss
   conda install pytorch torchvision torchaudio pytorch-cuda=YOUR_CUDA_VERSION -c pytorch -c nvidia
   pip install transformers accelerate gradio pillow
   pip install "huggingface_hub[cli]"
   ```

2. **Model Loading**: We have already hard-coded the huggingface log-in operation in the demo. The model will be downloaded to your local huggingface cache folder by default. If you want to specify the download directory,run `export HF_HOME="/path/to/your/directory"`.
  
### Gradio UI Demo

This mode provides a user-friendly web interface for intuitive multi-turn conversations and image interaction.

#### How to Run

Execute the following command in your terminal to start the Gradio service:

```bash
python vis_gradio.py
```

Once the service is running, you'll see a local URL (hardcoded as `http://127.0.0.1:7862`) and a public share URL. Open either URL in your browser to start using the demo.

#### Features

- **Chatbot**: Engage in multi-turn conversations in the main chat area.
- **Text Input**: Type your questions in the text box at the bottom.
- **Image Upload**: You can upload an image, and it will be sent to the model as part of the current turn's input.
- **Parameter Sliders**: Use the sliders on the right to adjust model parameters like **temperature**, **top_p**, and **max_length** in real-time.
- **Clear Chat**: Click the "Clear Chat" button to reset the conversation.

### Command-Line Demo

This mode has no UI and is ideal for backend services, scripts, or integration with other applications.

#### How to Run

Use the `python CerebraGloss_cli.py` command. The script supports two running modes:

1. **Single Call**

   Use command-line arguments to perform a single question-and-answer turn.

   - **Text only**:

     ```bash
     python CerebraGloss_cli.py -t "Hello, can you tell me what is eyeblink artifact?"
     ```

   - **Text and image**:

     ```bash
     python CerebraGloss_cli.py -t "Describe this eeg pattern." -i "figures/example_image.jpg"
     ```

   - **Specify generation parameters**: You can fine-tune the generation behavior by providing parameters.

     ```bash
     python CerebraGloss_cli.py -t "Describe this image." -i "figures/example_image.jpg" --temperature 1.0 --top_p 0.9
     ```

2. **Interactive Mode**

   Run the script without any arguments to enter an interactive session.

   ```bash
   python CerebraGloss_cli.py
   ```

   In this mode, you can interact with the model in a conversational style, supporting multi-turn dialogue.

   - **Input format**: `[Your text] | [Image path]`

   - **Example**:

     ```bash
     User: Describe this eeg pattern. | figures/example_image.jpg
     Model: ...
     User: Are there any eyeblink artifacts?
     Model: ...
     ```

   - **Exit**: Type `exit` to end the session.

#### Parameters

- `-t` or `--text`: User's input text (required).
- `-i` or `--image`: Path to the image file (optional).
- `--temperature`: Controls the randomness of the output; default is `0.95`.
- `--top_p`: Controls the diversity of the output; default is `0.7`.
- `--max_length`: Maximum number of tokens to generate; default is `1024`.