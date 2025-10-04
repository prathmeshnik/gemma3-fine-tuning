<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# okay now make a detailed .md file documentation mention wht each files does and how it does and how it is linked to each other,

Of course. Here is a detailed `README.md` file that documents the entire project, explaining what each file does, how it works, and how they are all interconnected.

***

# Fine-Tuning a Phi-3-Mini Model with Unsloth

This project demonstrates how to fine-tune the `unsloth/Phi-3-mini-4k-instruct-bnb-4bit` model using LoRA for a custom JSON extraction task. The process is broken down into modular Python scripts for training, testing, and exporting the model.

## File Structure

```
.
├── 📂 outputs/                  # Directory for saved model adapters after training
├── 📂 gguf_model/               # Directory for the exported GGUF model
├── 📜 main.py                    # Main script for training the model
├── 📜 test_model.py               # Script to test the fine-tuned model
├── 📜 export_gguf.py              # Script to convert the model to GGUF format
├── 📜 config.py                   # Configuration file for hyperparameters and paths
├── 📜 requirements.txt            # List of Python dependencies
├── 📜 json_extraction_dataset_500.json  # The training dataset
└── 📜 README.md                   # This documentation file
```


***

## File Descriptions

### 1. `config.py`

* **Purpose**: This file acts as a central hub for all configurations. Storing settings here makes it easy to change hyperparameters without modifying the main scripts.
* **How it Works**: It contains Python variables for model names, training parameters (like learning rate and batch size), and file paths.
* **Connections**: It is imported by `main.py`, `test_model.py`, and `export_gguf.py` to ensure that all parts of the workflow use the same settings consistently.


### 2. `requirements.txt`

* **Purpose**: To list all the Python packages required to run the project.
* **How it Works**: This is a standard text file that `pip` can read to install all necessary libraries in one command (`pip install -r requirements.txt`).
* **Connections**: It provides the foundational environment for all Python scripts. The scripts will fail to run if the libraries listed here are not installed.


### 3. `json_extraction_dataset_500.json`

* **Purpose**: This file contains the custom data used to fine-tune the model.
* **How it Works**: It's a JSON file structured as a list of examples, where each example has an `input` (the raw text/HTML) and an `output` (the desired structured JSON).
* **Connections**: This file is the primary input for the training process. The `load_dataset` function in `main.py` reads and processes this file.


### 4. `main.py`

* **Purpose**: The core script that handles the entire model fine-tuning process.
* **How it Works**: The script executes a sequence of steps:

1. **`check_gpu()`**: Verifies that a CUDA-enabled GPU is available.
2. **`load_dataset()`**: Reads `json_extraction_dataset_500.json`, applies a prompt template to each entry, and loads it into a `datasets.Dataset` object.
3. **`load_model()`**: Downloads the base `Phi-3-mini` model and its tokenizer using Unsloth's `FastLanguageModel`, which is optimized for speed and memory.
4. **`setup_lora()`**: Injects trainable LoRA (Low-Rank Adaptation) adapters into the model. This is a Parameter-Efficient Fine-Tuning (PEFT) method, meaning only the small adapter layers are trained, not the entire model.
5. **`train_model()`**: Uses the `SFTTrainer` from the `trl` library to perform supervised fine-tuning. It feeds the prepared dataset to the LoRA-equipped model and trains it.
* **Connections**:
    * **Reads from**: `json_extraction_dataset_500.json` and `config.py`.
    * **Writes to**: The `outputs/` directory, where it saves the trained LoRA adapter weights and training checkpoints.


### 5. `test_model.py`

* **Purpose**: To run a quick inference test on the fine-tuned model to see if it performs as expected.
* **How it Works**:

1. It loads the base `Phi-3-mini` model again.
2. It applies the trained LoRA adapters from the `outputs/` directory onto the base model.
3. It sets the model to inference mode using `FastLanguageModel.for_inference()`, which further optimizes it.
4. A sample prompt is defined, tokenized, and passed to `model.generate()` to produce a response.
* **Connections**:
    * **Reads from**: The `outputs/` directory (created by `main.py`) to load the LoRA adapters. It also reads from `config.py` for model settings.


### 6. `export_gguf.py`

* **Purpose**: To convert the fine-tuned model into GGUF format. GGUF is a file format that allows models to run efficiently on CPUs and various hardware through clients like `llama.cpp`.
* **How it Works**:

1. It loads the fine-tuned model by merging the LoRA adapters from the `outputs/` directory with the base model weights.
2. It uses Unsloth's built-in `save_pretrained_gguf()` function, which handles the quantization (e.g., `q4_k_m`) and conversion process.
* **Connections**:
    * **Reads from**: The `outputs/` directory to get the trained adapters.
    * **Writes to**: The `gguf_model/` directory, creating a new, self-contained `.gguf` file.

***

## Workflow: How the Files Connect

The project follows a linear workflow:

1. **Setup**: You start by installing all dependencies listed in **`requirements.txt`**. All scripts pull their configurations from **`config.py`**.
2. **Training (`main.py`)**: The training script takes your dataset (**`json_extraction_dataset_500.json`**) and uses it to train LoRA adapters for the base model. These new adapters, which represent the "learned knowledge," are saved in the **`outputs/`** directory.
3. **Testing (`test_model.py`)**: The testing script loads the adapters from **`outputs/`** to validate that the training was successful by running a sample prompt.
4. **Exporting (`export_gguf.py`)**: Finally, the export script takes the same adapters from **`outputs/`**, merges them into the base model, and converts the result into a portable **`.gguf`** file inside the **`gguf_model/`** directory.

## How to Run the Project

1. **Install Dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run Training**:
Place your dataset file in the root directory and run the main script.

```bash
python main.py
```

3. **Test the Model**:
Once training is complete and the `outputs/` directory exists, run the test script.

```bash
python test_model.py
```

4. **Export to GGUF**:
To create a GGUF file for inference, run the export script.

```bash
python export_gguf.py
```

