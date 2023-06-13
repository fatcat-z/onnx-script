# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any, List, Literal, Optional

from onnxscript.diagnostics.infra.sarif import (
    _address,
    _artifact,
    _conversion,
    _external_property_file_references,
    _graph,
    _invocation,
    _logical_location,
    _property_bag,
    _result,
    _run_automation_details,
    _special_locations,
    _thread_flow_location,
    _tool,
    _tool_component,
    _version_control_details,
    _web_request,
    _web_response,
)


@dataclasses.dataclass
class Run:
    """Describes a single run of an analysis tool, and contains the reported output of that run."""

    tool: _tool.Tool = dataclasses.field(metadata={"schema_property_name": "tool"})
    addresses: Optional[List[_address.Address]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "addresses"}
    )
    artifacts: Optional[List[_artifact.Artifact]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "artifacts"}
    )
    automation_details: Optional[
        _run_automation_details.RunAutomationDetails
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "automationDetails"})
    baseline_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "baselineGuid"}
    )
    column_kind: Optional[Literal["utf16CodeUnits", "unicodeCodePoints"]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "columnKind"}
    )
    conversion: Optional[_conversion.Conversion] = dataclasses.field(
        default=None, metadata={"schema_property_name": "conversion"}
    )
    default_encoding: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "defaultEncoding"}
    )
    default_source_language: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "defaultSourceLanguage"}
    )
    external_property_file_references: Optional[
        _external_property_file_references.ExternalPropertyFileReferences
    ] = dataclasses.field(
        default=None,
        metadata={"schema_property_name": "externalPropertyFileReferences"},
    )
    graphs: Optional[List[_graph.Graph]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphs"}
    )
    invocations: Optional[List[_invocation.Invocation]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "invocations"}
    )
    language: str = dataclasses.field(
        default="en-US", metadata={"schema_property_name": "language"}
    )
    logical_locations: Optional[List[_logical_location.LogicalLocation]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "logicalLocations"}
    )
    newline_sequences: List[str] = dataclasses.field(
        default_factory=lambda: ["\r\n", "\n"],
        metadata={"schema_property_name": "newlineSequences"},
    )
    original_uri_base_ids: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "originalUriBaseIds"}
    )
    policies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "policies"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    redaction_tokens: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "redactionTokens"}
    )
    results: Optional[List[_result.Result]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "results"}
    )
    run_aggregates: Optional[
        List[_run_automation_details.RunAutomationDetails]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "runAggregates"})
    special_locations: Optional[_special_locations.SpecialLocations] = dataclasses.field(
        default=None, metadata={"schema_property_name": "specialLocations"}
    )
    taxonomies: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "taxonomies"}
    )
    thread_flow_locations: Optional[
        List[_thread_flow_location.ThreadFlowLocation]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "threadFlowLocations"}
    )
    translations: Optional[List[_tool_component.ToolComponent]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "translations"}
    )
    version_control_provenance: Optional[
        List[_version_control_details.VersionControlDetails]
    ] = dataclasses.field(
        default=None, metadata={"schema_property_name": "versionControlProvenance"}
    )
    web_requests: Optional[List[_web_request.WebRequest]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webRequests"}
    )
    web_responses: Optional[List[_web_response.WebResponse]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webResponses"}
    )


# flake8: noqa
