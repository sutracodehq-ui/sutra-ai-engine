from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class ClickSignals(BaseModel):
    mouse_moves: int = Field(0, description="Number of mouse move events detected")
    scroll_depth: float = Field(0.0, description="Maximum scroll depth reached (0.0 to 1.0)")
    time_on_page_ms: int = Field(0, description="Time spent on page in milliseconds")
    touch_events: int = Field(0, description="Number of touch events detected (for mobile)")
    is_trusted_viewer: bool = Field(False, description="Whether the browser environment is trusted (e.g. not a headless browser)")

class ClickClientData(BaseModel):
    ua: str = Field(..., description="User Agent string")
    resolution: str = Field(..., description="Screen resolution (e.g. 1920x1080)")
    timezone: str = Field(..., description="Client timezone")
    fingerprint: str = Field(..., description="Unique browser fingerprint")
    language: str = Field(..., description="Client language")
    signals: ClickSignals = Field(default_factory=ClickSignals)

class ClickTrackRequest(BaseModel):
    ad_id: str = Field(..., description="Target Ad ID")
    tenant_id: int = Field(..., description="Tenant ID owning the ad")
    client_data: ClickClientData
    ip: Optional[str] = None
    referrer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ClickScoreReason(BaseModel):
    rule: str
    score_impact: int
    description: str

class ClickTrackResponse(BaseModel):
    click_id: str
    fraud_score: int = Field(..., ge=0, le=100, description="Overall fraud score (0-100)")
    is_blocked: bool
    action: str = Field(..., description="Recommended action: ALLOW, FLAG, or BLOCK")
    reasons: List[ClickScoreReason] = Field(default_factory=list)
    request_id: str

class ClickReportSummary(BaseModel):
    total_clicks: int
    blocked_clicks: int
    flagged_clicks: int
    fraud_percentage: float
    estimated_savings_usd: float
    top_fraud_reasons: List[str]
    period: str
