# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any, List, Literal, Optional

from onnxscript.diagnostics.infra.sarif import (
    _artifact_location,
    _attachment,
    _code_flow,
    _fix,
    _graph,
    _graph_traversal,
    _location,
    _message,
    _property_bag,
    _reporting_descriptor_reference,
    _result_provenance,
    _stack,
    _suppression,
    _web_request,
    _web_response,
)


@dataclasses.dataclass
class Result:
    """A result produced by an analysis tool."""

    message: _message.Message = dataclasses.field(metadata={"schema_property_name": "message"})
    analysis_target: Optional[_artifact_location.ArtifactLocation] = dataclasses.field(
        default=None, metadata={"schema_property_name": "analysisTarget"}
    )
    attachments: Optional[List[_attachment.Attachment]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "attachments"}
    )
    baseline_state: Optional[
        Literal["new", "unchanged", "updated", "absent"]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "baselineState"})
    code_flows: Optional[List[_code_flow.CodeFlow]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "codeFlows"}
    )
    correlation_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "correlationGuid"}
    )
    fingerprints: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "fingerprints"}
    )
    fixes: Optional[List[_fix.Fix]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "fixes"}
    )
    graph_traversals: Optional[List[_graph_traversal.GraphTraversal]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphTraversals"}
    )
    graphs: Optional[List[_graph.Graph]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "graphs"}
    )
    guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "guid"}
    )
    hosted_viewer_uri: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "hostedViewerUri"}
    )
    kind: Literal[
        "notApplicable", "pass", "fail", "review", "open", "informational"
    ] = dataclasses.field(default="fail", metadata={"schema_property_name": "kind"})
    level: Literal["none", "note", "warning", "error"] = dataclasses.field(
        default="warning", metadata={"schema_property_name": "level"}
    )
    locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "locations"}
    )
    occurrence_count: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "occurrenceCount"}
    )
    partial_fingerprints: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "partialFingerprints"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    provenance: Optional[_result_provenance.ResultProvenance] = dataclasses.field(
        default=None, metadata={"schema_property_name": "provenance"}
    )
    rank: float = dataclasses.field(default=-1.0, metadata={"schema_property_name": "rank"})
    related_locations: Optional[List[_location.Location]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "relatedLocations"}
    )
    rule: Optional[
        _reporting_descriptor_reference.ReportingDescriptorReference
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "rule"})
    rule_id: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "ruleId"}
    )
    rule_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "ruleIndex"}
    )
    stacks: Optional[List[_stack.Stack]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "stacks"}
    )
    suppressions: Optional[List[_suppression.Suppression]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "suppressions"}
    )
    taxa: Optional[
        List[_reporting_descriptor_reference.ReportingDescriptorReference]
    ] = dataclasses.field(default=None, metadata={"schema_property_name": "taxa"})
    web_request: Optional[_web_request.WebRequest] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webRequest"}
    )
    web_response: Optional[_web_response.WebResponse] = dataclasses.field(
        default=None, metadata={"schema_property_name": "webResponse"}
    )
    work_item_uris: Optional[List[str]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "workItemUris"}
    )


# flake8: noqa
