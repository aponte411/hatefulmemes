import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config.yaml")
def train(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print(type(cfg))
    print(cfg.model.vision_feature_dim)
if __name__ == "__main__":
    train()
