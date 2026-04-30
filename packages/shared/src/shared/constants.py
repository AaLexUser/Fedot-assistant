"""Single source of truth for time limits and AutoML engines exposed in UI and /config/options."""

AUTOML_ENGINE_OPTIONS = ["fedot", "autogluon"]

# Display label → seconds (Streamlit selector and REST /config/options)
TIME_LIMIT_MAPPING = {
    "5 мин": 300,
    "10 мин": 600,
    "30 мин": 1800,
    "1 ч": 3600,
    "2 ч": 7200,
    "4 ч": 14400,
}
