import yaml
from pathlib import Path

ROOT_DIR = Path(__file__).parent / '../..' # project root

class Pipeline:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.read_config(self.config_path)
        
    def parse_config(self):
        pass

    def read_config(self):
        with open(self.config_path) as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Unexpected {exc=}, {type(exc)=}")
                
        return config