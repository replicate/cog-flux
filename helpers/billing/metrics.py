import warnings

from cog import ExperimentalFeatureWarning, current_scope

warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)

# https://github.com/replicate/web/blob/main/replicate_web/metronome.py#L48-L65
# https://github.com/replicate/director/blob/fc47af0457a1eead08a2f6574ee06eb75c7f6c43/cog/types.go#L64
INTEGER_METRICS = {
    "audio_output_count",
    "character_input_count",
    "character_output_count",
    "generic_output_count",
    "image_output_count",
    "token_input_count",
    "token_output_count",
    "training_step_count",
    "video_output_count",
    "video_output_total_pixel_count",
}

FLOAT_METRICS = {
    "audio_output_duration_seconds",
    "unspecified_billing_metric",
    "video_output_duration_seconds",
}

STRING_METRICS = {
    "model_variant",
    "motion_mode",
    "resolution_target",
    "resolution_upscale_target",
}

BOOL_METRICS = {
    "with_audio",
}

ALL_METRICS = INTEGER_METRICS | FLOAT_METRICS | STRING_METRICS | BOOL_METRICS


def record_billing_metric(metric_name: str, value: float | int | str | bool) -> None:
    if metric_name not in ALL_METRICS:
        raise ValueError(f"Invalid metric name: {metric_name}")

    if metric_name in INTEGER_METRICS:
        if not isinstance(value, int):
            raise ValueError(
                f"Metric {metric_name} requires an integer value, got {type(value)}"
            )
        if value < 0:
            raise ValueError(f"Metric value must be non-negative, got {value}")
    elif metric_name in FLOAT_METRICS:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Metric {metric_name} requires a numeric value, got {type(value)}"
            )
        if value < 0:
            raise ValueError(f"Metric value must be non-negative, got {value}")
    elif metric_name in STRING_METRICS:
        if not isinstance(value, str):
            raise ValueError(
                f"Metric {metric_name} requires a string value, got {type(value)}"
            )
    elif metric_name in BOOL_METRICS:
        if not isinstance(value, bool):
            raise ValueError(
                f"Metric {metric_name} requires a boolean value, got {type(value)}"
            )

    current_scope().record_metric(metric_name, value)
