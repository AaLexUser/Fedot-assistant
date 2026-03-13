import difflib
import json
import logging
import re
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, Optional

import json_repair

from ..exceptions import OutputParserException

logger = logging.getLogger(__name__)


def _normalize_pathlike(value: str) -> str:
    """Normalize a path-like string to use forward slashes and no surrounding whitespace.

    Args:
        value: A path string potentially containing backslashes or whitespace.

    Returns:
        The normalized path string with forward slashes and stripped whitespace.

    Example:
        >>> _normalize_pathlike("  data\\\\train.csv  ")
        'data/train.csv'
    """
    return value.replace("\\", "/").strip()


def _resolve_valid_value(
    parsed_value: str,
    valid_values: Iterable[str],
) -> str | None:
    """Resolve a parsed value to one of the valid values using multiple matching strategies.

    Attempts to match the parsed_value to valid_values using increasingly lenient strategies:
    1. Exact match after normalization
    2. Suffix match (e.g., "data.csv" matches "path/to/data.csv")
    3. Basename match (filename only) if exactly one valid value matches
    4. Fuzzy string matching for typos or slight variations

    Args:
        parsed_value: The value to resolve, typically from LLM output.
        valid_values: An iterable of valid values to match against.

    Returns:
        The matching valid value from valid_values, or None if no match is found.

    Example:
        >>> valid = ["project/data/train.csv", "project/data/test.csv"]
        >>> _resolve_valid_value("train.csv", valid)
        'project/data/train.csv'
    """
    parsed_normalized = _normalize_pathlike(parsed_value).lstrip("./")
    valid_values_list = list(valid_values)

    for valid_value in valid_values_list:
        valid_normalized = _normalize_pathlike(str(valid_value))
        if valid_normalized == parsed_normalized:
            return str(valid_value)
        if valid_normalized.endswith(f"/{parsed_normalized}"):
            return str(valid_value)

    basename_matches = [
        str(valid_value)
        for valid_value in valid_values_list
        if PurePosixPath(_normalize_pathlike(str(valid_value))).name
        == PurePosixPath(parsed_normalized).name
    ]
    if len(basename_matches) == 1:
        return basename_matches[0]

    close_matches = difflib.get_close_matches(parsed_value, valid_values_list)
    if close_matches:
        return str(close_matches[0])

    return None


def parse_json(raw_reply: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from an LLM response, handling various formats and repairing malformed JSON.

    Handles JSON wrapped in markdown code blocks (with or without json language tag)
    and attempts to repair common JSON errors automatically.

    Args:
        raw_reply: The raw string response from an LLM that may contain JSON.

    Returns:
        A parsed dictionary if valid JSON is found, None otherwise.
        Malformed JSON will be repaired before parsing when possible.

    Example:
        >>> parse_json('{"key": "value"}')
        {'key': 'value'}
        >>> parse_json('```json\\n{"key": "value"}\\n```')
        {'key': 'value'}
    """

    def try_json_loads(data: str) -> Dict[str, Any] | None:
        try:
            repaired_json = json_repair.repair_json(
                data, ensure_ascii=False, return_objects=True
            )
            return repaired_json if repaired_json != "" else None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error: {e}")
            return None

    raw_reply = raw_reply.strip()
    # Case 1: Check if the JSON is enclosed in triple backticks
    json_match = re.search(r"\{.*\}|```(?:json)?\s*(.*?)```", raw_reply, re.DOTALL)
    if json_match:
        if json_match.group(1):
            reply_str = json_match.group(1).strip()
        else:
            reply_str = json_match.group(0).strip()
        reply = try_json_loads(reply_str)
        if reply is not None:
            return reply

    # Case 2: Assume the entire string is a JSON object
    return try_json_loads(raw_reply)


def check_json_values(
    parsed_json: Dict,
    valid_values: Optional[Iterable[str]],
    fallback_value: Optional[str],
):
    """Validate and resolve JSON values against a set of valid values.

    Iterates through JSON key-value pairs and resolves parsed values to valid values
    using fuzzy matching. If resolution fails and a fallback is provided, uses the
    fallback; otherwise raises an error.

    Args:
        parsed_json: The parsed JSON dictionary to validate. Modified in-place.
        valid_values: An iterable of valid values to match against. If None, validation is skipped.
        fallback_value: A default value to use if a value cannot be resolved.

    Returns:
        The validated/corrected parsed_json dictionary.

    Raises:
        ValueError: If a value cannot be resolved and no fallback_value is provided.

    Example:
        >>> check_json_values({"col": "train"}, ["train.csv", "test.csv"], None)
        {"col": "train.csv"}
    """
    if valid_values is not None:
        for key, parsed_value in parsed_json.items():
            # Currently only support single parsed value
            if isinstance(parsed_value, list) and len(parsed_value) == 1:
                parsed_value = parsed_value[0]
            if isinstance(parsed_value, str):
                resolved_value = _resolve_valid_value(parsed_value, valid_values)
            else:
                logger.warning(
                    f"Unrecognized parsed value: {parsed_value} for key {key} parsed by the LLM. "
                    f"It has type: {type(parsed_value)}."
                )
                resolved_value = None

            if resolved_value is None:
                if fallback_value:
                    logger.warning(
                        f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM. "
                        f"Will use default value: {fallback_value}."
                    )
                    parsed_json[key] = fallback_value
                else:
                    raise ValueError(
                        f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM."
                    )
            else:
                parsed_json[key] = resolved_value
    return parsed_json


def parse_and_check_json(
    raw_reply: str,
    expected_keys: Iterable[str],
    valid_values: Optional[Iterable[str]] = None,
    fallback_value: Optional[str] = None,
):
    """Parse JSON from LLM output and validate it has required keys and valid values.

    Orchestrates the full JSON parsing and validation workflow: parse the raw reply,
    extract expected keys, validate values against valid_values, and handle errors.

    Args:
        raw_reply: The raw string response from an LLM.
        expected_keys: Keys that must be present in the parsed JSON.
        valid_values: Optional set of valid values for the JSON fields.
        fallback_value: Default value to use if a value cannot be resolved.

    Returns:
        A validated dictionary with only the expected_keys.

    Raises:
        OutputParserException: If JSON parsing fails, required keys are missing,
                               or values cannot be resolved without a fallback.

    Example:
        >>> parse_and_check_json('{"col": "train"}', ["col"], ["train.csv"])
        {"col": "train.csv"}
    """
    if json_obj := parse_json(raw_reply):
        for key in expected_keys:
            if key not in json_obj:
                error = f"Got invalid return object. Expected key `{key}` "
                f"to be present, but got {json_obj}"
                logging.error(error)
                raise OutputParserException(error)
        json_obj = {key: json_obj[key] for key in expected_keys}
        try:
            check_json_values(json_obj, valid_values, fallback_value)
        except ValueError as e:
            raise OutputParserException(e)
        return json_obj
    raise OutputParserException("JSON decoding error or JSON not found in output")


def get_outer_columns(all_columns, num_columns_each_end=10):
    """Get the first and last N columns from a list of columns.

    Returns the full list if it's small enough, otherwise returns the outer
    columns (first N and last N) to fit all columns in limited space.

    Args:
        all_columns: An iterable of column names or identifiers.
        num_columns_each_end: Number of columns to include from each end. Defaults to 10.

    Returns:
        A list containing either all columns (if small) or first/last N columns.

    Example:
        >>> get_outer_columns(list(range(50)), 5)
        [0, 1, 2, 3, 4, 45, 46, 47, 48, 49]
    """
    if len(all_columns) <= num_columns_each_end * 2:
        return list(all_columns)
    return all_columns[:num_columns_each_end] + all_columns[-num_columns_each_end:]
