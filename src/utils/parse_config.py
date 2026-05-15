import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the training config file")
    parser.add_argument(
        "--peft-config", type=str, default=None,
        help="Path to a PEFT config file (LoRA / QLoRA). "
             "When provided, the peft block is merged into the main config. "
             "Omit to run full fine-tuning.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.peft_config is not None:
        with open(args.peft_config, "r") as f:
            peft_config = yaml.safe_load(f)
        config["peft"] = peft_config

    print(f"Config: {config}")
    return config