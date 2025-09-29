import sys
from .deepspeed.helper import parse_ds_config, get_ds_checkpoint_version

HOST_CACHE_SIZE="host_cache_size"
HOST_CACHE_SIZE_DEFAULT=0
CKPT_PARSER_THREADS="parser_threads"
CKPT_PARSER_THREADS_DEFAULT=4
FAST_CACHE_INIT="fast_cache_init"
FAST_CACHE_INIT_DEFAULT=False
PIN_HOST_CACHE="pin_host_cache"
PIN_HOST_CACHE_DEFAULT=True

# Partial update support
PARTIAL_UPDATE_MODE="partial_update_mode"
PARTIAL_UPDATE_MODE_DEFAULT=False
FULL_CHECKPOINT_INTERVAL="full_checkpoint_interval"
FULL_CHECKPOINT_INTERVAL_DEFAULT=500  # Every 500 steps (practical)
PARTIAL_CHECKPOINT_INTERVAL="partial_checkpoint_interval"  # NEW
PARTIAL_CHECKPOINT_INTERVAL_DEFAULT=100  # Every 100 steps (practical)
UPDATE_BUDGET_RATIO="update_budget_ratio"
UPDATE_BUDGET_RATIO_DEFAULT=0.3

# Gradient tracking for layer selection
GRADIENT_TRACKING_ENABLED="gradient_tracking_enabled"
GRADIENT_TRACKING_ENABLED_DEFAULT=True

# Warmup configuration
WARMUP_STEPS="warmup_steps"
WARMUP_STEPS_DEFAULT=100

SUPPORTED_CONFIG_CLASSES = tuple(["dict", "OrderedDict", "DeepSpeedConfig"])

IS_DEEPSPEED_ENABLED = False

def get_config_type(config) -> str:
    config_class = str(type(config))
    if 'DeepSpeedConfig' in config_class:
        return 'DeepSpeedConfig'
    elif 'dict' in config_class or 'OrderedDict' in config_class:
        return 'dict'
    raise Exception(f"Config class ({config_class}) not supported. Please use from {SUPPORTED_CONFIG_CLASSES}.")

def parse_config(config) -> dict:
    global IS_DEEPSPEED_ENABLED
    config_class = get_config_type(config)
    result = {
        HOST_CACHE_SIZE: HOST_CACHE_SIZE_DEFAULT,
        CKPT_PARSER_THREADS: CKPT_PARSER_THREADS_DEFAULT,
        FAST_CACHE_INIT: FAST_CACHE_INIT_DEFAULT,
        PIN_HOST_CACHE: PIN_HOST_CACHE_DEFAULT,
        PARTIAL_UPDATE_MODE: PARTIAL_UPDATE_MODE_DEFAULT,
        FULL_CHECKPOINT_INTERVAL: FULL_CHECKPOINT_INTERVAL_DEFAULT,
        PARTIAL_CHECKPOINT_INTERVAL: PARTIAL_CHECKPOINT_INTERVAL_DEFAULT,  # NEW
        UPDATE_BUDGET_RATIO: UPDATE_BUDGET_RATIO_DEFAULT,
        GRADIENT_TRACKING_ENABLED: GRADIENT_TRACKING_ENABLED_DEFAULT,
        WARMUP_STEPS: WARMUP_STEPS_DEFAULT,
    }
    
    if config_class == "DeepSpeedConfig":
        checkpointing_config = parse_ds_config(config)
        IS_DEEPSPEED_ENABLED = True
    elif config_class == "dict":
        checkpointing_config = config         

    for k, _ in result.items():
        if k in checkpointing_config:
            result[k] = checkpointing_config[k]
    return result
    
def get_checkpoint_version(ckpt_path, last_version=-1) -> int:
    version = last_version + 1
    if IS_DEEPSPEED_ENABLED:
        version = get_ds_checkpoint_version(ckpt_path, last_version)
    return version