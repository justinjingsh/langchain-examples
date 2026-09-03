"""
Shared factory for the Bedrock embeddings client, used by every sample that
needs to embed text (embeddings.py, embeddings_chroma.py,
embedding_chroma_retrieval.py, embedding_chroma_persistence.py).

Pulled out of embeddings.py so those other samples don't have to import a
module named after one specific demo just to construct a client.
"""

from langchain_aws import BedrockEmbeddings

from config import Config


def build_embeddings_client(config: Config) -> BedrockEmbeddings:
    """Construct a BedrockEmbeddings client from the resolved app config."""
    return BedrockEmbeddings(
        model_id=config.model.embedding_model_id,
        region_name=config.aws.region,
        aws_access_key_id=config.aws.access_key_id,
        aws_secret_access_key=config.aws.secret_access_key,
        aws_session_token=config.aws.session_token,
    )
