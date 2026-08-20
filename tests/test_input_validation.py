"""Tests for shared.python.input_validation."""

import pytest

from shared.python.input_validation import (
    sanitize_prompt_input,
    validate_email,
    validate_number_input,
    validate_text_input,
    validate_url,
)


class TestValidateNumberInput:
    def test_valid_number(self):
        assert validate_number_input("5", min_val=1, max_val=20) == 5

    def test_strips_whitespace(self):
        assert validate_number_input("  7  ") == 7

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="between"):
            validate_number_input("0", min_val=1, max_val=20)

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError, match="between"):
            validate_number_input("21", min_val=1, max_val=20)

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            validate_number_input("abc")

    def test_boundaries_are_inclusive(self):
        assert validate_number_input("1", min_val=1, max_val=20) == 1
        assert validate_number_input("20", min_val=1, max_val=20) == 20

    def test_negative_range_supported(self):
        assert validate_number_input("-5", min_val=-10, max_val=0) == -5

    def test_non_string_raises_with_field_name(self):
        with pytest.raises(ValueError, match="age"):
            validate_number_input(None, field_name="age")

    def test_error_message_includes_field_name(self):
        with pytest.raises(ValueError, match="quantity"):
            validate_number_input("nope", field_name="quantity")


class TestValidateTextInput:
    def test_valid_text(self):
        assert validate_text_input("Hello World", max_length=100) == "Hello World"

    def test_trims_result(self):
        assert validate_text_input("  padded  ") == "padded"

    def test_empty_not_allowed_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_text_input("   ")

    def test_empty_allowed_returns_empty(self):
        assert validate_text_input("   ", allow_empty=True) == ""

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            validate_text_input("x" * 11, max_length=10)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            validate_text_input("ab", min_length=5)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            validate_text_input(None)

    def test_none_allowed_returns_empty(self):
        assert validate_text_input(None, allow_empty=True) == ""

    def test_max_length_boundary_accepted(self):
        assert validate_text_input("x" * 10, max_length=10) == "x" * 10

    def test_error_messages_use_field_name(self):
        with pytest.raises(ValueError, match="recipe name"):
            validate_text_input("", field_name="recipe name")


class TestSanitizePromptInput:
    def test_plain_text_unchanged(self):
        assert sanitize_prompt_input("Hello, world!") == "Hello, world!"

    def test_removes_template_injection(self):
        result = sanitize_prompt_input("Hello {{system}} world")
        assert "{{" not in result
        assert "}}" not in result

    def test_removes_variable_substitution(self):
        result = sanitize_prompt_input("value ${danger} here")
        assert "${" not in result

    def test_removes_script_tags(self):
        result = sanitize_prompt_input("hi <script>alert(1)</script> there")
        assert "<script" not in result.lower()

    def test_removes_javascript_url(self):
        result = sanitize_prompt_input("click javascript:alert(1)")
        assert "javascript:" not in result.lower()

    def test_empty_returns_empty(self):
        assert sanitize_prompt_input("") == ""

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            sanitize_prompt_input("x" * 11, max_length=10)

    def test_only_invalid_characters_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            sanitize_prompt_input("{{a}}")

    def test_removes_control_characters(self):
        assert sanitize_prompt_input("ab\x00c\x07d") == "abcd"

    def test_normalizes_whitespace(self):
        assert sanitize_prompt_input("  hello \n\t  world  ") == "hello world"

    def test_strict_mode_drops_unsafe_characters(self):
        result = sanitize_prompt_input("safe [text] `backtick` {brace}", strict=True)
        assert "[" not in result
        assert "]" not in result
        assert "`" not in result
        assert "{" not in result
        assert "safe text backtick brace" == result

    def test_strict_mode_keeps_basic_punctuation(self):
        assert sanitize_prompt_input("Hello, world! (ok?)", strict=True) == "Hello, world! (ok?)"

    def test_none_returns_empty(self):
        assert sanitize_prompt_input(None) == ""


class TestValidateEmail:
    def test_valid_email_lowercased(self):
        assert validate_email("User@Example.COM") == "user@example.com"

    @pytest.mark.parametrize(
        "bad", ["notanemail", "missing@domain", "@nolocal.com", "spaces in@x.com"]
    )
    def test_invalid_email_raises(self, bad):
        with pytest.raises(ValueError):
            validate_email(bad)


class TestValidateUrl:
    def test_valid_https(self):
        assert validate_url("https://example.com/path") == "https://example.com/path"

    def test_http_rejected_when_https_required(self):
        with pytest.raises(ValueError):
            validate_url("http://example.com")

    def test_http_allowed_when_not_required(self):
        assert validate_url("http://example.com", require_https=False) == "http://example.com"

    def test_garbage_raises(self):
        with pytest.raises(ValueError):
            validate_url("not a url")

    def test_garbage_raises_when_https_not_required(self):
        with pytest.raises(ValueError, match="Invalid URL"):
            validate_url("ftp://example.com", require_https=False)

    def test_strips_surrounding_whitespace(self):
        assert validate_url("  https://example.com  ") == "https://example.com"

    def test_url_with_whitespace_inside_raises(self):
        with pytest.raises(ValueError):
            validate_url("https://example.com/a b")
