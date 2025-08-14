import uuid
from pydantic import BaseModel, Field

class Persona(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    background: str
    expertise: str
    viewpoints: str
    communication_style: str = Field(default="Professional")
    assertiveness: int = Field(default=5)  # Scale 1-10
    cooperation: int = Field(default=5)    # Scale 1-10
    color: str = Field(default="#3366cc")    # For avatar generation
    is_user: bool = Field(default=False)
