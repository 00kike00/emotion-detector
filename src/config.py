from pathlib import Path
import torch

# 1. BASE DIRECTORY
# This locates the 'emotion-detector' root folder automatically from src/
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. DATA PATHS
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Dataset Specific Paths
FER_DIR = RAW_DIR / "FER2013"
GOEMOTIONS_DIR = RAW_DIR / "GoEmotions" / "data"
MELD_DIR = RAW_DIR / "MELD"
RAVDESS_DIR = RAW_DIR / "RAVDESS" / "Audio_Speech_Actors_01-24"

# 3. MODEL & CHECKPOINT PATHS
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# 4. PLOTS & FIGURES
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# 5. GLOBAL SETTINGS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 48          # Standard for FER2013
BATCH_SIZE = 128       # Larger batch size for faster training, adjust based on GPU memory
NUM_CLASSES = 7        # Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral

# 6. LLM SETTINGS
LLM_MODEL = "gemma3:1b"
LLM_PATH = BASE_DIR / "src" / "llm_wrapper"
LLM_PROMPT_PATH = LLM_PATH / "system_prompt.txt"
KBS_PROMPT_PATH = LLM_PATH / "system_prompt_kbs.txt"

# 7. APSO SETTINGS (For Hyperparameter Optimization)
APSO_ITERATIONS = 10
APSO_PARTICLES = 10
PROXY_EPOCHS = 5       # How many epochs to run for each particle test

# 8. KNOWLEDGE BASED SYSTEM
KBS_PATH = BASE_DIR / "src" / "kbs"

if __name__ == "__main__":
    print(f"--- Configuration Verified ---")
    print(f"Project Root: {BASE_DIR}")
    print(f"Using Device: {DEVICE}")
    print(f"FER2013 Path: {FER_DIR} (Exists: {FER_DIR.exists()})")