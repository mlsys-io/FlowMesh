from enum import StrEnum


class TaskType(StrEnum):
    INFERENCE = "inference"
    RAG = "rag"
    DIFFUSION = "diffusion"
    API = "api"
    SFT = "sft"
    LORA_SFT = "lora_sft"
    PPO = "ppo"
    DPO = "dpo"
    IMAGE_CLASSIFICATION_TRAINING = "image_classification_training"
    ECHO = "echo"
    AGENT = "agent"
    DATA_PROFILING = "data_profiling"
    DATA_RETRIEVAL = "data_retrieval"
    EMBEDDING = "embedding"
    SSH = "ssh"
    OMNI_TEXT2IMAGE = "omni_text2image"
    OMNI_TEXT2SPEECH = "omni_text2speech"
    OMNI_TEXT2AUDIO = "omni_text2audio"
    OMNI_TEXT2GENERAL = "omni_text2general"
    SERVE = "serve"
