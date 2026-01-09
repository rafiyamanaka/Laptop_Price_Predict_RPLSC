from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "predictor"

    def ready(self) -> None:
        # Load artifacts if available (avoid crashing management commands
        # when model hasn't been trained yet).
        from .runtime import load_artifacts

        load_artifacts()
