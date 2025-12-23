"""Shared utilities for vllm-score worker and client."""


def estimate_token_count(data: dict) -> int:
    """
    Estimate token count from text_1 and text_2 arrays using word count * 1.4.
    Use a tokenizer for accuracy
    Used for both benchmark and per-request workload calculation.

    Args:
        data: Dictionary containing 'text_1' and/or 'text_2' lists of strings

    Returns:
        Estimated token count (minimum 1)
    """
    total_words = 0
    for text_list in [data.get("text_1", []), data.get("text_2", [])]:
        for text in text_list:
            if isinstance(text, str):
                total_words += len(text.split())
    return max(int(total_words * 1.4), 1)  # Ensure at least 1 to avoid zero workload
