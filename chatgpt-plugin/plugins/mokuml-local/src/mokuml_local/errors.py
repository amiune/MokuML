"""Only these fixed messages can cross the model-visible error boundary."""

MESSAGES = {
    "ready": "Ready. Input records and identifying details stay on this computer.",
    "no_dataset": "Choose a dataset in the local window first.",
    "invalid_selection": "Choose columns from the current dataset and keep identifiers separate from the outcome and features.",
    "file_invalid": "Choose a valid UTF-8 CSV or Excel .xlsx file with unique, nonempty column names.",
    "file_too_large": "Choose a spreadsheet no larger than 50 MB.",
    "too_many_rows": "Select no more than 100,000 records. No rows were silently omitted.",
    "too_many_columns": "Select no more than 2,000 columns.",
    "empty_data": "The selected table has no records.",
    "database_failed": "The database could not be read. Check access, TLS settings, table permissions, and database locks in the local window.",
    "picker_failed": "Choose an existing SQLite database using the local macOS file picker.",
    "worksheet_required": "Choose the worksheet in the local window.",
    "models_missing": "Prepare the prediction models in the local window. Text features also need the local Qwen model.",
    "models_invalid": "Model files failed verification. Prepare the models again in the local window.",
    "setup_running": "The local model download is running.",
    "setup_failed": "Model setup failed. Check the internet connection and free disk space, then try again.",
    "insufficient_data": "More labeled history is needed for independent training, validation and testing. Add records or simplify the task.",
    "insufficient_classes": "Classification requires at least two outcomes in the training records. Add more historical examples.",
    "invalid_outcome": "Regression needs finite numeric outcomes. Choose classification for categories.",
    "text_too_long": "Some records exceed the local encoder's token or memory budget. Shorten text or select fewer input columns.",
    "resource_limit": "This dataset needs more local memory than the configured budget. Use fewer rows or text columns.",
    "inference_failed": "The local model could not complete this run. Close other applications or use a smaller dataset and try again.",
    "busy": "A prediction is already running. Wait or cancel it in the local window.",
    "unknown_run": "That run is no longer available. Prepare a prediction using the current dataset.",
    "cancelled": "The run was cancelled and temporary data was released.",
    "released": "The dataset and its prediction results were released.",
    "window_opened": "The private local window was opened in your system browser.",
    "window_failed": "The local window could not be opened. Check the system browser and try again.",
    "internal_error": "The operation could not complete. Retry using the local window.",
    "small_data": "Validation uses a small amount of historical data; treat results as an early estimate.",
    "limited_classes": "There are too few validation examples for a reliable ROC AUC measurement.",
    "cpu_retry": "The same models ran on the CPU after graphics acceleration failed.",
    "below_baseline": "The model did not improve on the simple baseline in this validation split.",
    "no_source": "Connect a database or import data in the private local window first.",
    "invalid_plan": "The task could not be compiled. Use current catalog IDs, supported expressions and a clear prediction unit.",
    "stale_plan": "The source structure changed. Inspect it again and prepare a new prediction plan.",
    "unsafe_query": "The generated query did not pass the required read-only AST checks.",
    "invalid_split": "The requested split would mix entities, future information, or incomplete outcomes. Add history or correct the grouping and time definitions.",
    "ambiguous_grain": "The prepared table contains duplicate prediction units. Aggregate joined records or select the correct identifier columns.",
    "no_compatible_models": "No installed model supports this task within the local resource limits. Prepare compatible models or reduce the dataset.",
    "scoring_mismatch": "The new-case data does not match the prediction features. Map the scoring columns in the local window or prepare a corrected plan.",
    "interval_unavailable": "There are too few independent calibration examples for a 90% prediction interval.",
}


class SafeError(Exception):
    def __init__(self, code: str):
        self.code = code if code in MESSAGES else "internal_error"
        super().__init__(MESSAGES[self.code])
