from here.application.controller import HereApplicationController


def create_default_controller() -> HereApplicationController:
    """Build the production controller without importing CLI or UI modules."""

    return HereApplicationController()
