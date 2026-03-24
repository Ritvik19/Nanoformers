from dotenv import load_dotenv

from src.utils.parse_config import parse_args


def load_config():
    args = parse_args()
    load_dotenv(args["env_path"])
    return args
