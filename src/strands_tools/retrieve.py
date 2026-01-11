"""
Amazon Bedrock Knowledge Base retrieval tool for Strands Agent.

This module provides functionality to perform semantic search against Amazon Bedrock
Knowledge Bases, enabling natural language queries against your organization's documents.
It uses vector-based similarity matching to find relevant information and returns results
ordered by relevance score.

Key Features:
1. Semantic Search:
   • Vector-based similarity matching
   • Relevance scoring (0.0-1.0)
   • Score-based filtering

2. Advanced Configuration:
   • Custom result limits
   • Score thresholds
   • Regional support
   • Multiple knowledge bases

3. Response Format:
   • Sorted by relevance
   • Includes metadata
   • Source tracking
   • Score visibility

Usage with Strands Agent:
```python
from strands import Agent
from strands_tools import retrieve

agent = Agent(tools=[retrieve])

# Basic search with default knowledge base and region
results = agent.tool.retrieve(text="What is the STRANDS SDK?")

# Search with metadata enabled for source information
results = agent.tool.retrieve(
    text="What is the STRANDS SDK?",
    enableMetadata=True
)

# Advanced search with custom parameters
results = agent.tool.retrieve(
    text="deployment steps for production",
    numberOfResults=5,
    score=0.7,
    knowledgeBaseId="custom-kb-id",
    region="us-east-1",
    enableMetadata=True,
    retrieveFilter={
        "andAll": [
            {"equals": {"key": "category", "value": "security"}},
            {"greaterThan": {"key": "year", "value": "2022"}}
        ]
    }
)
```

See the retrieve function docstring for more details on available parameters and options.
"""

import os
from typing import Any, Dict, List

import boto3
from botocore.config import Config as BotocoreConfig
from strands import tool


def filter_results_by_score(results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    """
    Filter results based on minimum score threshold.

    This function takes the raw results from a knowledge base query and removes
    any items that don't meet the minimum relevance score threshold.

    Args:
        results: List of retrieval results from Bedrock Knowledge Base
        min_score: Minimum score threshold (0.0-1.0). Only results with scores
            greater than or equal to this value will be returned.

    Returns:
        List of filtered results that meet or exceed the score threshold
    """
    return [result for result in results if result.get("score", 0.0) >= min_score]


def format_results_for_display(results: List[Dict[str, Any]], enable_metadata: bool = False) -> str:
    """
    Format retrieval results for readable display.

    This function takes the raw results from a knowledge base query and formats
    them into a human-readable string with scores, document IDs, and content.
    Optionally includes metadata when enabled.

    Args:
        results: List of retrieval results from Bedrock Knowledge Base
        enable_metadata: Whether to include metadata in the formatted output (default: False)

    Returns:
        Formatted string containing the results in a readable format, including score,
        document ID, optional metadata, and content.
    """
    if not results:
        return (
            "No results found above score threshold.\n"
            "POSSIBLE CAUSES:\n"
            "1. Score threshold too high (try lowering 'score' parameter)\n"
            "2. No documents match the query\n"
            "3. Knowledge base is empty or not indexed\n"
            "4. Query terms don't match document content"
        )

    formatted = []
    for result in results:
        # Extract document location - handle both s3Location and customDocumentLocation
        location = result.get("location", {})
        doc_id = "Unknown"
        if "customDocumentLocation" in location:
            doc_id = location["customDocumentLocation"].get("id", "Unknown")
        elif "s3Location" in location:
            # Extract meaningful part from S3 URI
            doc_id = location["s3Location"].get("uri", "")
        score = result.get("score", 0.0)
        formatted.append(f"\nScore: {score:.4f}")
        formatted.append(f"Document ID: {doc_id}")

        content = result.get("content", {})
        if content and isinstance(content.get("text"), str):
            text = content["text"]
            formatted.append(f"Content: {text}\n")

        # Add metadata if enabled and present
        if enable_metadata:
            metadata = result.get("metadata")
            if metadata:
                formatted.append(f"Metadata: {metadata}")

    return "\n".join(formatted)


@tool
def retrieve(
    text: str,
    numberOfResults: int = 10,
    knowledgeBaseId: str = None,
    region: str = None,
    score: float = None,
    profile_name: str = None,
    enableMetadata: bool = None,
    retrieveFilter: Dict = None,
) -> Dict:
    """Retrieve relevant knowledge from Amazon Bedrock Knowledge Base.

    This tool uses Amazon Bedrock Knowledge Bases to perform semantic search against your
    organization's documents. It returns results sorted by relevance score, with the ability
    to filter results that don't meet a minimum score threshold.

    How It Works:
    1. The provided query text is sent to Amazon Bedrock Knowledge Base
    2. The service performs vector-based semantic search against indexed documents
    3. Results are returned with relevance scores (0.0-1.0) indicating match quality
    4. Results below the minimum score threshold are filtered out
    5. Remaining results are formatted for readability and returned

    Common Usage Scenarios:
    - Answering user questions from product documentation
    - Finding relevant information in company policies
    - Retrieving context from technical manuals
    - Searching for relevant sections in research papers
    - Looking up information in legal documents

    Args:
        text: The query text to search for in the knowledge base
        numberOfResults: Maximum number of results to return (default: 10)
        knowledgeBaseId: The ID of the knowledge base to query (default: from KNOWLEDGE_BASE_ID env)
        region: AWS region where the knowledge base is located (default: us-west-2 or AWS_REGION env)
        score: Minimum relevance score threshold (default: 0.4 or MIN_SCORE env)
        profile_name: Optional AWS profile name to use
        enableMetadata: Whether to include metadata in the response (default: false or RETRIEVE_ENABLE_METADATA_DEFAULT env)
        retrieveFilter: Optional filter to apply to the retrieval results

    Returns:
        Dictionary containing status and response content

    Notes:
        - The knowledge base ID can be set via the KNOWLEDGE_BASE_ID environment variable
        - The AWS region can be set via the AWS_REGION environment variable
        - The minimum score threshold can be set via the MIN_SCORE environment variable
        - Results are automatically filtered based on the minimum score threshold
        - AWS credentials must be configured properly for this tool to work
    """
    # Handle environment variable defaults
    default_knowledge_base_id = os.getenv("KNOWLEDGE_BASE_ID")
    default_aws_region = os.getenv("AWS_REGION", "us-west-2")
    default_min_score = float(os.getenv("MIN_SCORE", "0.4"))
    default_enable_metadata = os.getenv("RETRIEVE_ENABLE_METADATA_DEFAULT", "false").lower() == "true"

    # Use defaults if not provided
    if knowledgeBaseId is None:
        knowledgeBaseId = default_knowledge_base_id
    if region is None:
        region = default_aws_region
    if score is None:
        score = default_min_score
    if enableMetadata is None:
        enableMetadata = default_enable_metadata

    try:
        # Validate required parameters
        if not knowledgeBaseId:
            raise ValueError(
                "ERROR: knowledgeBaseId is required but not provided.\n"
                "SOLUTION 1: Set environment variable: export KNOWLEDGE_BASE_ID='your-kb-id'\n"
                "SOLUTION 2: Pass parameter: retrieve(text='query', knowledgeBaseId='your-kb-id')\n"
                "NOTE: You can find your Knowledge Base ID in the AWS Bedrock console"
            )

        config = BotocoreConfig(user_agent_extra="strands-agents-retrieve")
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            bedrock_agent_runtime_client = session.client(
                "bedrock-agent-runtime", region_name=region, config=config
            )
        else:
            bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=region, config=config)

        # Default retrieval configuration
        retrieval_config = {"vectorSearchConfiguration": {"numberOfResults": numberOfResults}}

        if retrieveFilter:
            try:
                if _validate_filter(retrieveFilter):
                    retrieval_config["vectorSearchConfiguration"]["filter"] = retrieveFilter
            except ValueError as e:
                return {
                    "status": "error",
                    "content": [{"text": str(e)}],
                }

        # Perform retrieval
        response = bedrock_agent_runtime_client.retrieve(
            retrievalQuery={"text": text}, knowledgeBaseId=knowledgeBaseId, retrievalConfiguration=retrieval_config
        )

        # Get and filter results
        all_results = response.get("retrievalResults", [])
        filtered_results = filter_results_by_score(all_results, score)

        # Format results for display with optional metadata
        formatted_results = format_results_for_display(filtered_results, enableMetadata)

        # Return success with formatted results
        return {
            "status": "success",
            "content": [
                {"text": f"Retrieved {len(filtered_results)} results with score >= {score}:\n{formatted_results}"}
            ],
        }

    except Exception as e:
        # Return error with helpful details
        error_msg = str(e)

        # Provide specific help for common errors
        if "NoCredentialsError" in error_msg or "Unable to locate credentials" in error_msg:
            error_msg = (
                f"AWS credentials not configured.\n"
                f"SOLUTION: Configure AWS credentials using one of:\n"
                f"1. AWS CLI: aws configure\n"
                f"2. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
                f"3. IAM role (if running on AWS)\n"
                f"Original error: {error_msg}"
            )
        elif "AccessDeniedException" in error_msg:
            error_msg = (
                f"Access denied to Bedrock Knowledge Base.\n"
                f"SOLUTION: Ensure your AWS credentials have the required permissions:\n"
                f"- bedrock:Retrieve permission for the knowledge base\n"
                f"- Check the knowledge base ID is correct: {knowledgeBaseId}\n"
                f"Original error: {error_msg}"
            )
        elif "ResourceNotFoundException" in error_msg:
            error_msg = (
                f"Knowledge Base not found.\n"
                f"SOLUTION: Verify:\n"
                f"1. Knowledge Base ID is correct: {knowledgeBaseId}\n"
                f"2. Region is correct: {region}\n"
                f"3. Knowledge Base exists and is active\n"
                f"Original error: {error_msg}"
            )

        return {
            "status": "error",
            "content": [{"text": f"Error during retrieval: {error_msg}"}],
        }


# A simple validator to check filter is in valid shape
def _validate_filter(retrieve_filter):
    """Validate the structure of a retrieveFilter."""
    try:
        if not isinstance(retrieve_filter, dict):
            raise ValueError("retrieveFilter must be a dictionary")

        # Valid operators according to AWS Bedrock documentation
        valid_operators = [
            "equals",
            "greaterThan",
            "greaterThanOrEquals",
            "in",
            "lessThan",
            "lessThanOrEquals",
            "listContains",
            "notEquals",
            "notIn",
            "orAll",
            "andAll",
            "startsWith",
            "stringContains",
        ]

        # Validate each operator in the filter
        for key, value in retrieve_filter.items():
            if key not in valid_operators:
                raise ValueError(f"Invalid operator: {key}")

            # Validate operator value structure
            if key in ["orAll", "andAll"]:  # Both orAll and andAll require arrays
                if not isinstance(value, list):
                    raise ValueError(f"Value for '{key}' operator must be a list")
                if len(value) < 2:  # Both require minimum 2 items
                    raise ValueError(f"Value for '{key}' operator must contain at least 2 items")
                for sub_filter in value:
                    _validate_filter(sub_filter)
            else:
                if not isinstance(value, dict):
                    raise ValueError(f"Value for '{key}' operator must be a dictionary")
        return True
    except Exception as e:
        raise Exception(f"Unexpected error while validating retrieve filter: {str(e)}") from e
