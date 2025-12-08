
from .config import RefinerLMConfig
from .train import train

def main(run, config):

    cfg = RefinerLMConfig(**config)

    train(run, cfg)

if __name__ == "__main__":
    # parse input json from first arg
    # run = None
    import sys
    import json

    with open(sys.argv[1], "r") as f:
        input_json = json.load(f)
    cfg = input_json.get("input", {}).get("config", {})
    print("Running RefinerLM main with config:", cfg)
    main(None, cfg)