"""
Sample code that verifies AWS Bedrock credentials/model access are working,
by sending one trivial prompt through `ChatBedrockConverse` and printing the
response (or the error, if credentials/model access are misconfigured).

Run directly with: uv run python check_connection.py
"""

from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage

from config import Config, load_config


def main() -> None:
    print("Check bedrock connection")
    config = load_config()
    # Print the resolved config (but not the actual secret values) so it's
    # obvious at a glance whether env vars/.env were picked up correctly.
    print(f"AWS region: {config.aws.region}")
    print(f"credentials loaded: {config.aws.access_key_id is not None}")
    print(f"Embedding model id: {config.model.embedding_model_id}")
    print(f"Bedrock model id: {config.model.bedrock_model_id}")

    check_bedrock_connection(config)


def check_bedrock_connection(config: Config) -> None:
    """Send a trivial prompt to Bedrock to confirm credentials and model access work."""
    model_id = config.model.bedrock_model_id
    # ChatBedrockConverse uses Bedrock's newer "Converse" API, which gives a
    # consistent request/response shape across different foundation models
    # (Anthropic, Amazon, Meta, etc.) instead of each model having its own format.
    llm = ChatBedrockConverse(
        model=model_id,
        region_name=config.aws.region,
        aws_access_key_id=config.aws.access_key_id,
        aws_secret_access_key=config.aws.secret_access_key,
        aws_session_token=config.aws.session_token,
    )

    try:
        # A single HumanMessage is the simplest possible chat request - just
        # enough to prove the round trip to Bedrock and back works.
        response = llm.invoke([HumanMessage(content="What is the capital city of China")])
        print(f"Bedrock call succeeded (model={model_id}).")
        print("Response:")
        print(f"{response.content!r}")
    except ClientError as e:
        # ClientError covers AWS-side failures (bad credentials, no access to
        # this model, wrong region, throttling, etc.) - catching it here means
        # this script reports the problem instead of crashing with a traceback.
        print(f"Bedrock call failed (model={model_id}): {e}")

if (__name__ == "__main__"):
    main()
