# **Emotion Recognition**

**##** Requirements
**-** Python 3.11+
**-** SWI-Prolog 9.x — https://www.swi-prolog.org/download/stable
**-** Ollama — https://ollama.com/download
**-** Pull the model: **`ollama pull gemma3:1b`**

**##** Installation
**``**bash** **pip **install** -r requirements.txt** **``

**##** Model Files
Place the following files in the **`models/`** directory:
**-**`final_vision_expert_best_ft.pth`
**-**`final_text_expert_best.pth`
**-**`final_manager_best.pth`
**-**`face_detection/deploy.prototxt`
**-**`face_detection/res10_300x300_ssd_iter_140000.caffemodel`

**##** Running the Demo
Make sure Ollama is running first

Then launch the KBS demo:
**``**bash** **python src/main_demo.py** **``

**##** Project Structure
**-**`src/architectures/` — VisionNet, RoBERTa-BiLSTM, ManagerNet
**-**`src/kbs/` — Prolog knowledge base and Python wrapper
**-**`src/llm_wrapper/` — Ollama integration
**-**`src/main_demo.py` — PyQt6 KBS demo
