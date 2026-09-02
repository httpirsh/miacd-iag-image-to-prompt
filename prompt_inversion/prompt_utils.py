MAX_PROMPT_TOKENS = 75


def limit_prompt_tokens(prompt, tokenizer=None, max_tokens=MAX_PROMPT_TOKENS):
    """
    Keep prompts within the CLIP text-encoder token limit.
    Uses the LCM tokenizer when available; otherwise falls back to word-based shortening.
    """
    prompt = " ".join(prompt.strip().split())

    if tokenizer is None:
        return " ".join(prompt.split()[:max_tokens])

    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=max_tokens,
        return_tensors=None,
    )

    return tokenizer.decode(
        tokenized["input_ids"],
        skip_special_tokens=True,
    ).strip()


def unique_preserve_order(items):
    seen = set()
    unique_items = []
    for item in items:
        item = " ".join(str(item).strip().split())
        if item and item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def unique_prompt_entries(entries):
    """Deduplicate prompt entries while preserving the first prompt_type."""
    seen = set()
    output = []
    for entry in entries:
        if isinstance(entry, str):
            prompt = " ".join(entry.strip().split())
            prompt_type = "unspecified"
        else:
            prompt = " ".join(str(entry.get("prompt", "")).strip().split())
            prompt_type = entry.get("prompt_type", "unspecified")

        if prompt and prompt not in seen:
            seen.add(prompt)
            output.append({"prompt": prompt, "prompt_type": prompt_type})
    return output


def join_prompt_parts(*parts):
    """Join prompt fragments while removing empty parts."""
    return ", ".join(
        part.strip(" ,")
        for part in parts
        if isinstance(part, str) and part.strip()
    )


def unpack_prompt_entry(entry):
    """Support both plain prompt strings and prompt metadata dictionaries."""
    if isinstance(entry, dict):
        return entry.get("prompt", ""), entry.get("prompt_type", "unspecified")
    return str(entry), "unspecified"


def make_entry(prompt, prompt_type, tokenizer=None, max_tokens=MAX_PROMPT_TOKENS):
    prompt = limit_prompt_tokens(prompt, tokenizer=tokenizer, max_tokens=max_tokens)
    return {"prompt": prompt, "prompt_type": prompt_type}
