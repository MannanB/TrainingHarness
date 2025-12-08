
from .config import RefinerLMConfig
from .train import train

def main(run, config):

    cfg = RefinerLMConfig(**config)

    train(run, cfg)