"""Custom exceptions for the server module."""

from __future__ import annotations


class JobCancelled(Exception):
    """Raised inside ``run_pipeline`` when a job's cancel token is set."""
