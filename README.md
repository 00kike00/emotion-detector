# MERA — Multimodal Emotion Recognition Agent

Multimodal emotion recognition system combining facial expression and text analysis through a GRU-based manager network and a knowledge-based system (KBS) for emotion-aware conversational interaction

## Requirements

**-** Python 3.13.7
**-** SWI-Prolog 9.x — https://www.swi-prolog.org/download/stable
**-** Ollama — https://ollama.com/download
**-** Pull the model: **`ollama pull gemma3:1b`**

## Installation

**``**bash** **pip **install** -r requirements.txt** **``

## Model Files

Place the following files in the **`models/`** directory:
**-**`final_vision_expert_best_ft.pth`
**-**`final_text_expert_best.pth`
**-**`final_manager_best.pth`
**-**`face_detection/deploy.prototxt`
**-**`face_detection/res10_300x300_ssd_iter_140000.caffemodel`

The following link leads to a dirve folder with all the models stored inside.

[https://drive.google.com/drive/folders/1uPf722_mq5eRSeUXQPj0XQue-gWUS8sm?usp=sharing](https://drive.google.com/drive/folders/1uPf722_mq5eRSeUXQPj0XQue-gWUS8sm?usp=sharing)

## Running the Demo

Make sure Ollama is running first

Then launch the KBS demo:
**``**bash** **python src/main_demo.py** **``

## Project Structure

**-**`src/data_pipeline/` — Dataset loading, preprocessing and augmentation
**-**`src/fine_tuning/` — Expert finetuning on MELD
**-**`src/inference/` — Inference scripts for each model
**-**`src/model_eval/` — Evaluation scripts
**-**`src/model_trainers/` — Training loops for VisionNet, RoBERTa-BiLSTM and ManagerNet
**-**`src/optimization/` — APSO hyperparameter search
**-**`src/config.py` — Global configuration
**-**`src/main_demo.py` — PyQt6 KBS demo
