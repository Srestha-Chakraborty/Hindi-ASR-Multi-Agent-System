def tag_english_words(text: str) -> str:
    from post_processing.cleanup import CleanupConfig, detect_english_words

    return str(detect_english_words(text, CleanupConfig(), return_metadata=False))
