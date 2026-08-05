"""
Gemini API-based maintenance analyzer for the Aircraft Maintenance Platform.

Sends the PDF manual directly to Gemini (native PDF understanding) along with
engineering analytics JSON, and asks for a structured JSON maintenance report.
This replaces the Bedrock/Nova Pro version because Gemini has a generous free
tier and does not require AWS account-level model activation.

Requirements:
    pip install google-genai

Before running:
    1. Get a free Gemini API key: https://aistudio.google.com/apikey
    2. Put it in backend/.env as GEMINI_API_KEY=your-key-here
"""

from __future__ import annotations

import json
import importlib
import logging
import os
from pathlib import Path
from typing import Any

try:
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
except ImportError as exc:
    raise ImportError(
        "google-genai is required. Install it with: pip install google-genai"
    ) from exc

logger = logging.getLogger(__name__)

# JSON schema describing the required output shape. Gemini is constrained to
# return data matching this structure (equivalent to the schema previously
# described in the Bedrock prompt).
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "aircraft": {"type": "string"},
        "aircraft_model": {"type": "string"},
        "health_status": {
            "type": "string",
            "enum": ["SAFE", "MONITOR", "MAINTENANCE REQUIRED", "GROUND AIRCRAFT"],
        },
        "risk_level": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"],
        },
        "safe_for_next_flight": {"type": "boolean"},
        "final_flight_decision": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": [
                        "CLEARED_TO_FLY",
                        "FLY_WITH_MONITORING",
                        "MAINTENANCE_REQUIRED_BEFORE_FLIGHT",
                        "GROUND_AIRCRAFT",
                    ],
                },
                "can_fly_now": {"type": "boolean"},
                "ui_statement": {"type": "string"},
                "required_before_next_flight": {"type": "string"},
                "decision_rationale": {"type": "string"},
            },
            "required": [
                "decision",
                "can_fly_now",
                "ui_statement",
                "required_before_next_flight",
                "decision_rationale",
            ],
        },
        "overall_summary": {"type": "string"},
        "threshold_violations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "observed_value": {"type": "string"},
                    "manual_threshold": {"type": "string"},
                    "severity": {
                        "type": "string",
                        "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"],
                    },
                    "manual_reference": {"type": "string"},
                    "explanation": {"type": "string"},
                },
            },
        },
        "root_cause": {
            "type": "object",
            "properties": {
                "most_likely_cause": {"type": "string"},
                "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                "manual_reference": {"type": "string"},
            },
        },
        "maintenance_actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "integer"},
                    "action": {"type": "string"},
                    "reason": {"type": "string"},
                    "manual_reference": {"type": "string"},
                },
            },
        },
        "inspection_checklist": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "inspection_item": {"type": "string"},
                    "acceptance_criteria": {"type": "string"},
                    "manual_reference": {"type": "string"},
                },
            },
        },
        "work_order": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "aircraft_id": {"type": "string"},
                "work_order_type": {
                    "type": "string",
                    "enum": ["INSPECTION", "REPAIR", "MONITORING", "GROUNDING"],
                },
                "priority": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                },
                "tasks": {"type": "array", "items": {"type": "string"}},
                "required_parts_or_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "estimated_maintenance_category": {"type": "string"},
            },
        },
        "confidence": {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "rationale": {"type": "string"},
                "missing_information": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": [
        "aircraft",
        "aircraft_model",
        "health_status",
        "risk_level",
        "safe_for_next_flight",
        "final_flight_decision",
        "overall_summary",
        "threshold_violations",
        "root_cause",
        "maintenance_actions",
        "inspection_checklist",
        "work_order",
        "confidence",
    ],
}


class AircraftMaintenanceAnalyzerGemini:
    """
    Generate AI maintenance reports using the Gemini API and a manual PDF.
    """

    def __init__(
        self,
        model_name: str,
        manual_pdf_path: str | Path,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4_000,
    ) -> None:
        self.model_name = model_name
        self.manual_pdf_path = Path(manual_pdf_path)
        self.temperature = temperature
        self.max_tokens = max_tokens

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set the GEMINI_API_KEY environment "
                "variable or pass api_key explicitly."
            )
        self.client = genai.Client(api_key=key)

    def load_manual_bytes(self) -> bytes:
        """Load the complete aircraft maintenance manual PDF as bytes."""
        if not self.manual_pdf_path.exists():
            raise FileNotFoundError(
                f"Maintenance manual not found: {self.manual_pdf_path}"
            )
        if self.manual_pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Maintenance manual must be a PDF file. Received: {self.manual_pdf_path}"
            )

        logger.info("Loading maintenance manual: %s", self.manual_pdf_path)
        return self.manual_pdf_path.read_bytes()

    def build_prompt(self, engineering_analytics: dict[str, Any]) -> str:
        """Build the model prompt from engineering analytics JSON."""
        analytics_json = json.dumps(
            engineering_analytics,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
You are a Senior Aircraft Maintenance Engineer.

Your task is to generate a professional aircraft maintenance engineering report
using two inputs:

1. Engineering Analytics JSON provided below.
2. The attached internal Aircraft Maintenance Manual PDF.

Important rules:
- Compare every engineering parameter in the analytics JSON against the
  thresholds, safe operating limits, risk matrix, decision trees, inspection
  procedures, failure modes, and maintenance actions defined in the manual.
- Use only thresholds and maintenance procedures found in the attached manual.
- Do not invent thresholds, limits, failure modes, or maintenance actions.
- If a required threshold or procedure is unavailable in the manual, state that
  explicitly in the JSON output.
- Prioritize maintenance actions when multiple actions apply.
- Determine whether the aircraft status is one of:
  SAFE, MONITOR, MAINTENANCE REQUIRED, GROUND AIRCRAFT.
- Produce a final flight decision for the operations dashboard.

Engineering Analytics JSON:
{analytics_json}
""".strip()

    def analyze(self, engineering_analytics: dict[str, Any]) -> dict[str, Any]:
        """Generate a structured AI maintenance report from analytics and PDF."""
        prompt = self.build_prompt(engineering_analytics)
        manual_bytes = self.load_manual_bytes()

        contents = [
            types.Part.from_bytes(data=manual_bytes, mime_type="application/pdf"),
            prompt,
        ]

        try:
            logger.info("Invoking Gemini model: %s", self.model_name)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                ),
            )
        except Exception as exc:
            logger.exception("Gemini invocation failed")
            raise RuntimeError(
                f"Unable to invoke Gemini model '{self.model_name}': {exc}"
            ) from exc

        response_text = getattr(response, "text", None)

        if not response_text:
            finish_reason = None
            safety_info = None
            if getattr(response, "candidates", None):
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, "finish_reason", None)
                safety_info = getattr(candidate, "safety_ratings", None)
            logger.error(
                "Gemini returned no text. finish_reason=%s safety_ratings=%s raw_response=%s",
                finish_reason,
                safety_info,
                response,
            )
            raise RuntimeError(
                f"Gemini returned an empty response (finish_reason={finish_reason}). "
                "This usually means the output was cut off by max_tokens, blocked by "
                "a safety filter, or the schema could not be satisfied. Check server "
                "logs for the full raw response."
            )

        return self._parse_json_response(response_text)

    @staticmethod
    def _parse_json_response(response_text: str) -> dict[str, Any]:
        """Parse and validate the JSON-only model response."""
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            logger.error("Model returned non-JSON response: %s", response_text)
            raise ValueError("Gemini response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Gemini response JSON must be an object")

        return parsed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    base_dir = Path(__file__).resolve().parents[2]
    manual_pdf_path = base_dir / "data" / "AeroTech_ATX200_Maintenance_Manual.pdf"

    if not manual_pdf_path.exists():
        raise FileNotFoundError(f"Maintenance manual not found: {manual_pdf_path}")

    engineering_json = {
        # paste your engineering analytics JSON here for standalone testing
    }

    analyzer = AircraftMaintenanceAnalyzerGemini(
        model_name="gemini-3.6-flash",
        manual_pdf_path=manual_pdf_path,
        temperature=0.2,
        max_tokens=8000,
    )

    result = analyzer.analyze(engineering_json)
    print(json.dumps(result, indent=2))