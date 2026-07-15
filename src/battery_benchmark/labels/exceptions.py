"""Exceptions for SOC label generation."""

from __future__ import annotations


class SOCLabelError(Exception):
    """Base exception for SOC label generation failures."""


class InvalidRecordRoleForSOCError(SOCLabelError):
    """Raised when ``record_role`` is not eligible for SOC labelling."""


class UnsupportedSOCFileTypeError(SOCLabelError):
    """Raised when ``file_type`` is not supported by the requested calculator."""


class InvalidSOCInputError(SOCLabelError):
    """Raised when input data fail SOC calculator validation."""
