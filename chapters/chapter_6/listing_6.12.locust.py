# First start a streaming server
# python chapters/chapter_6/listing_6.3.streaming.py

# Run the locust test
# locust -f chapters/chapter_6/listing_6.12.locust.py

import time
from locust import HttpUser, task, events

# Create a CSV file to store custom stats
stat_file = open("stats.csv", "w")
stat_file.write("Latency,TTFT,TPS\n")


class StreamUser(HttpUser):
    @task
    def generate(self):
        # Initiate test
        token_count = 0
        start = time.time()

        # Make Request
        with self.client.post(
            "/generate",
            data='{"prompt": "Salt Lake City is a"}',
            catch_response=True,
            stream=True,
        ) as response:
            first_response = time.time()
            for line in response.iter_lines(decode_unicode=True):
                token_count += 1

        # Finish and calculate stats
        end = time.time()
        latency = end - start
        ttft = first_response - start
        tps = token_count / (end - first_response)

        # Save stats
        stat_file.write(f"{latency},{ttft},{tps}\n")


# Close stats file when Locust quits
@events.quitting.add_listener
def close_stats_file(environment):
    stat_file.close()
