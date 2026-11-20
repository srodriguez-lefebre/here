from here.application.controller import HereApplicationController
from here.memory.service import MemoryService


def create_default_controller(
    *, memory_service: MemoryService | None = None
) -> HereApplicationController:
    """Build the production controller without importing CLI or UI modules."""

    return HereApplicationController(memory_service=memory_service)
