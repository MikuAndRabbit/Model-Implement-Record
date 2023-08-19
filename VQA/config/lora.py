import os
import yaml
from peft import LoraConfig


LORA_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lora.yaml')
with open(LORA_CONFIG_PATH) as f:
    lora_config = yaml.safe_load(f)

VIT_LORA_CONFIG = LoraConfig(**lora_config['vit'])
BERT_LORA_CONFIG = LoraConfig(**lora_config['bert'])
