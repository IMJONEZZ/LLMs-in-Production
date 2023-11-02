import os
import pandas as pd

import whylogs as why
from langkit import llm_metrics
from datasets import load_dataset

OUTPUT_DIR = "logs"


class LoggingApp:
    def __init__(self):
        """
        Sets up a logger that collects profiles and writes them
        locally every 5 minutes. By setting the schema with langkit
        we get useful metrics for LLMs.
        """
        self.logger = why.logger(
            mode="rolling",
            interval=5,
            when="M",
            base_name="profile_",
            schema=llm_metrics.init(),
        )
        self.logger.append_writer("local", base_dir=OUTPUT_DIR)

    def close(self):
        self.logger.close()

    def consume(self, text):
        self.logger.log(text)


def driver(app):
    """Driver function to run the app manually"""
    data = load_dataset(
        "shahules786/OA-cornell-movies-dialog",
        split="train",
        streaming=True,
    )
    data = iter(data)
    for text in data:
        app.consume(text)


if __name__ == "__main__":
    # Run app manually
    app = LoggingApp()
    driver(app)
    app.close()

    # Prevent truncation of columns
    pd.set_option("display.max_columns", None)

    # Get the first profile and show results
    all_files = [
        f for f in os.listdir(OUTPUT_DIR) if f.startswith("profile_")
    ]
    path = os.path.join(OUTPUT_DIR, all_files[0])
    result_view = why.read(path).view()
    print(result_view.to_pandas().head())
# column        cardinality/est  cardinality/lower_1  cardinality/upper_1
# conversation      1158.936483          1144.135654          1174.106326
# ...
# column        counts/inf  counts/n  counts/nan  counts/null
# conversation           0      1160           0            0
# ...
# column        udf/flesch_reading_ease:cardinality/est
# conversation                               425.514743
# ...
# column        udf/jailbreak_similarity:cardinality/est
# conversation                               1172.226702
# ...
# column        udf/toxicity:types/string  udf/toxicity:types/tensor
# conversation                          0                          0
