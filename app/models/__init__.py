"""All SQLAlchemy models — imported here for Alembic auto-detection."""

from app.models.base import Base  # noqa: F401
from app.models.tenant import Tenant  # noqa: F401
from app.models.ai_conversation import AiConversation  # noqa: F401
from app.models.ai_task import AiTask  # noqa: F401
from app.models.agent_feedback import AgentFeedback  # noqa: F401
from app.models.voice_profile import VoiceProfile  # noqa: F401
from app.models.agent_training_data import AgentTrainingData  # noqa: F401
from app.models.token_usage_log import TokenUsageLog  # noqa: F401
from app.models.agent_optimization import AgentOptimization  # noqa: F401
