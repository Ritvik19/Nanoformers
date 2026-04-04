import os
import sys

# Allow `python src/cli/train_*.py ...` from any working directory.
# This file lives at `src/cli/...`, so the repo root is two levels up.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.training.contrastive_learning.image_text_sigmoid_contrastive.pipeline import main


if __name__ == "__main__":
    main()
