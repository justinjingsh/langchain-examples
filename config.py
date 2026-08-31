"""
Loads app configuration (AWS credentials/region and Bedrock model ids) from
environment variables / a local .env file into typed, immutable dataclasses.

Keeping config in one place like this means the rest of the codebase (e.g.
check_connection.py) never calls os.getenv() directly - it just asks for a
Config object and gets typed fields with sensible defaults already applied.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Reads a .env file in the working directory (if present) and populates
# os.environ, so os.getenv() calls below can pick up local dev credentials
# without them being committed to source control. See .env.example.
load_dotenv()

# Environment variable names, as constants so a typo becomes a NameError
# instead of a silently-missing config value.
AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
AWS_SESSION_TOKEN = "AWS_SESSION_TOKEN"
AWS_REGION = "AWS_REGION"

BEDROCK_MODEL_ID = "BEDROCK_MODEL_ID"
EMBEDDING_MODEL_ID = "EMBEDDING_MODEL_ID"

# Fallback model ids used when the corresponding env var isn't set, so the
# samples work out of the box without every field being configured.
DEFAULT_BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
DEFAULT_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


@dataclass(frozen=True)
class AWSConfig:
    """AWS credentials and region used to call Bedrock.

    All fields are optional (str | None) because in some environments (e.g.
    an EC2 instance role, or the AWS CLI's own credential chain) these are
    resolved automatically rather than passed explicitly - callers should
    handle None rather than assuming credentials are always set here.
    """
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None
    region: str | None


@dataclass(frozen=True)
class ModelConfig:
    """Bedrock model ids used for chat/completion and for embeddings."""
    bedrock_model_id: str
    embedding_model_id: str


@dataclass(frozen=True)
class Config:
    """Top-level app config, grouping AWS and model settings together."""
    aws: AWSConfig
    model: ModelConfig


def load_aws_config() -> AWSConfig:
    """Read AWS credentials/region from environment variables.

    Returns None for any variable that isn't set, rather than raising -
    validation of "is this actually usable" is left to the caller (e.g.
    the Bedrock client will raise its own clear error if credentials are
    missing when a call is attempted).
    """
    return AWSConfig(
        access_key_id=os.getenv(AWS_ACCESS_KEY_ID),
        secret_access_key=os.getenv(AWS_SECRET_ACCESS_KEY),
        session_token=os.getenv(AWS_SESSION_TOKEN),
        region=os.getenv(AWS_REGION),
    )


def load_model_config() -> ModelConfig:
    """Read Bedrock model ids from environment variables, falling back to
    the DEFAULT_* constants above when not set."""
    return ModelConfig(
        bedrock_model_id=os.getenv(BEDROCK_MODEL_ID, DEFAULT_BEDROCK_MODEL_ID),
        embedding_model_id=os.getenv(EMBEDDING_MODEL_ID, DEFAULT_EMBEDDING_MODEL_ID),
    )


def load_config() -> Config:
    """Build the full app Config by combining AWS and model config.

    This is the single entry point the rest of the app should call.
    """
    return Config(
        aws=load_aws_config(),
        model=load_model_config(),
    )
