# event.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import ClassVar, Optional, Any, Dict
from enum import Enum # Keep Enum from app.py context if EventType is defined there

# Assume EventType Enum is defined (e.g., in app.py or a shared types file)
class EventType(Enum):
    STATUS = "status"
    LOG = "log"
    ERROR = "error"
    RESULT = "result" # Event indicating the result file is ready
    COMPLETE = "complete" # Task finished (success or failure signaled by context)

class Event(BaseModel):
    """Base event model for SSE communication."""
    type: EventType
    message: str
    data: Optional[Dict[str, Any]] = None # Optional structured data payload
    _id: int = -1
    _timestamp: Optional[str] = None

    @property
    def id(self) -> int:
        return self._id

    @property
    def timestamp(self) -> Optional[str]:
        return self._timestamp

    class Config:
        allow_population_by_field_name = True # If you use aliases
        # Pydantic v2 doesn't need exclude_none=True in model_dump by default usually
        # json_encoders = { datetime: lambda dt: dt.isoformat() } # Not needed with string field

    def to_dict(self) -> dict:
        dump = self.model_dump(exclude={'_id', '_timestamp'}, exclude_none=True)
        dump['id'] = self.id
        dump['timestamp'] = self._timestamp
        return dump

# --- Specific Event Types ---

class LogEvent(Event):
    """Event specifically for log messages."""
    type: EventType = EventType.LOG
    level: Optional[str] = None # e.g., INFO, WARNING, ERROR

class ErrorEvent(Event):
    """Event for reporting errors."""
    type: EventType = EventType.ERROR
    details: Optional[str] = None # Optional traceback or more info

class StatusEvent(Event):
    """Event for general status updates."""
    type: EventType = EventType.STATUS

class ResultEvent(Event):
    """Event indicating the result file is ready."""
    type: EventType = EventType.RESULT
    filename: str # The name of the result file

class CompleteEvent(Event):
    """Event indicating the entire task process has finished."""
    type: EventType = EventType.COMPLETE
    success: bool # True if successful, False if ended due to error