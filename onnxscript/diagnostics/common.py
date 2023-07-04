"""Diagnostic components for ONNXScript."""
from __future__ import annotations

import contextlib
import gzip
from collections.abc import Generator
from typing import List, Optional, Type

import onnxscript

from onnxscript.diagnostics import infra
from onnxscript.diagnostics.infra import formatter, sarif
from onnxscript.diagnostics.infra.sarif import version as sarif_version


class ONNXScriptDiagnostic(infra.Diagnostic):
    """Base class for all onnxscript diagnostics.

    This class is used to represent all onnxscript diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    python_call_stack: Optional[infra.Stack] = None

    def __init__(
        self,
        *args,
        frames_to_skip: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.python_call_stack = self.record_python_call_stack(
            frames_to_skip=frames_to_skip
        )


class ONNXScriptDiagnosticEngine:
    """ONN Script diagnostic engine.

    The only purpose of creating this class instead of using `DiagnosticContext` directly
    is to provide a background context for `diagnose` calls inside exporter.

    By design, one `torch.onnx.export` call should initialize one diagnostic context.
    All `diagnose` calls inside exporter should be made in the context of that export.
    However, since diagnostic context is currently being accessed via a global variable,
    there is no guarantee that the context is properly initialized. Therefore, we need
    to provide a default background context to fallback to, otherwise any invocation of
    exporter internals, e.g. unit tests, will fail due to missing diagnostic context.
    This can be removed once the pipeline for context to flow through the exporter is
    established.
    """

    contexts: List[infra.DiagnosticContext]
    _background_context: infra.DiagnosticContext

    def __init__(self) -> None:
        self.contexts = []
        self._background_context = infra.DiagnosticContext(
            name="onnsxcript",
            version="1.14",
        )

    @property
    def background_context(self) -> infra.DiagnosticContext:
        return self._background_context

    def create_diagnostic_context(
        self,
        name: str,
        version: str,
        options: Optional[infra.DiagnosticOptions] = None,
        diagnostic_type: Type[infra.Diagnostic] = infra.Diagnostic,
    ) -> infra.DiagnosticContext:
        """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
        if options is None:
            options = infra.DiagnosticOptions()
        context = infra.DiagnosticContext(name, version, options)
        self.contexts.append(context)
        return context

    def clear(self):
        """Clears all diagnostic contexts."""
        self.contexts.clear()
        self._background_context.diagnostics.clear()

    def to_json(self) -> str:
        return formatter.sarif_to_json(self.sarif_log())

    def dump(self, file_path: str, compress: bool = False) -> None:
        """Dumps the SARIF log to a file."""
        if compress:
            with gzip.open(file_path, "wt") as f:
                f.write(self.to_json())
        else:
            with open(file_path, "w") as f:
                f.write(self.to_json())

    def sarif_log(self):
        log = sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            runs=[context.sarif() for context in self.contexts],
        )

        log.runs.append(self._background_context.sarif())
        return log


engine = ONNXScriptDiagnosticEngine()
_context = engine.background_context


def diagnose(
    rule: infra.Rule,
    level: infra.Level,
    message: Optional[str] = None,
    frames_to_skip: int = 2,
    **kwargs,
) -> ONNXScriptDiagnostic:
    """Creates a diagnostic and record it in the global diagnostic context.

    This is a wrapper around `context.record` that uses the global diagnostic context.
    """
    # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
    diagnostic = ONNXScriptDiagnostic(
        rule, level, message, frames_to_skip=frames_to_skip, **kwargs
    )
    onnxscript_context().add_diagnostic(diagnostic)
    return diagnostic


def onnxscript_context() -> infra.DiagnosticContext:
    global _context
    return _context
