import os
from itertools import cycle

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# vLLM model configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = 18000
MODEL_LOG_FILE             = '/var/log/portal/vllm.log'
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-specific log messages
MODEL_LOAD_LOG_MSG = [
    "Application startup complete.",
]

MODEL_ERROR_LOG_MSGS = [
    "INFO exited: vllm",
    "RuntimeError: Engine",
    "Traceback (most recent call last):"
]

MODEL_INFO_LOG_MSGS = [
    '"message":"Download'
]

# Benchmark text pairs for scoring - factual Q&A pairs that should score high
# Each tuple is (query, document) representing a relevant match
BENCHMARK_TEXT_PAIRS = [
    ("What is the capital of France?", "Paris is the capital and largest city of France."),
    ("How does photosynthesis work?", "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare around 1594."),
    ("What causes earthquakes?", "Earthquakes occur when tectonic plates suddenly slip past each other."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How many bones are in the human body?", "An adult human body contains 206 bones."),
    ("What is the chemical formula for water?", "The chemical formula for water is H2O."),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("What is the tallest mountain on Earth?", "Mount Everest is the tallest mountain on Earth at 8,849 meters."),
    ("What year did World War II end?", "World War II ended in 1945."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("How many continents are there?", "There are seven continents on Earth."),
    ("What is the freezing point of water?", "Water freezes at 0 degrees Celsius or 32 degrees Fahrenheit."),
    ("Who invented the telephone?", "Alexander Graham Bell invented the telephone in 1876."),
    ("What is the capital of Japan?", "Tokyo is the capital city of Japan."),
    ("How many planets are in our solar system?", "There are eight planets in our solar system."),
    ("What is DNA?", "DNA is a molecule that carries genetic instructions for living organisms."),
    ("Who was the first person to walk on the moon?", "Neil Armstrong was the first person to walk on the moon in 1969."),
    ("What is the largest mammal?", "The blue whale is the largest mammal on Earth."),
    ("What is the boiling point of water?", "Water boils at 100 degrees Celsius or 212 degrees Fahrenheit."),
    ("Who discovered penicillin?", "Alexander Fleming discovered penicillin in 1928."),
    ("What is the longest river in the world?", "The Nile is the longest river in the world at about 6,650 kilometers."),
    ("What is the hardest natural substance?", "Diamond is the hardest natural substance on Earth."),
    ("Who wrote the theory of relativity?", "Albert Einstein wrote the theory of relativity."),
    ("What is the smallest country in the world?", "Vatican City is the smallest country in the world."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("What is the capital of Australia?", "Canberra is the capital city of Australia."),
    ("What element does the symbol Au represent?", "Au is the chemical symbol for gold."),
    ("Who invented the light bulb?", "Thomas Edison invented the practical incandescent light bulb."),
    ("What is the Great Wall of China?", "The Great Wall of China is a series of fortifications stretching over 13,000 miles."),
    ("What is the currency of the United Kingdom?", "The currency of the United Kingdom is the British Pound Sterling."),
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What is the largest desert on Earth?", "The Sahara is the largest hot desert on Earth."),
    ("Who wrote Hamlet?", "Hamlet was written by William Shakespeare."),
    ("What is the atomic number of carbon?", "Carbon has an atomic number of 6."),
    ("What is the capital of Germany?", "Berlin is the capital city of Germany."),
    ("How many teeth does an adult human have?", "An adult human typically has 32 teeth."),
    ("What is the largest bird in the world?", "The ostrich is the largest living bird in the world."),
    ("Who painted the Sistine Chapel ceiling?", "Michelangelo painted the Sistine Chapel ceiling."),
    ("What is the main component of the Sun?", "The Sun is primarily composed of hydrogen."),
    ("What year was the Declaration of Independence signed?", "The Declaration of Independence was signed in 1776."),
    ("What is the capital of Brazil?", "Brasilia is the capital city of Brazil."),
    ("How many legs does a spider have?", "A spider has eight legs."),
    ("What is the largest continent?", "Asia is the largest continent by both area and population."),
    ("Who developed the polio vaccine?", "Jonas Salk developed the polio vaccine in 1955."),
    ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe."),
    ("What is the capital of Canada?", "Ottawa is the capital city of Canada."),
    ("How fast does sound travel?", "Sound travels at approximately 343 meters per second in air at room temperature."),
    ("What is the largest organ in the human body?", "The skin is the largest organ in the human body."),
]

# Create a cycling iterator for benchmark pairs
_benchmark_iterator = cycle(BENCHMARK_TEXT_PAIRS)


def request_parser(request):
    """Parse incoming request to extract data."""
    data = request
    if request.get("input") is not None:
        data = request.get("input")
    return data


def score_benchmark_generator() -> dict:
    """Generate benchmark payload for /score endpoint with sequential text pairs."""
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    # Get the next text pair from the cycling iterator
    query, document = next(_benchmark_iterator)

    benchmark_data = {
        "model": model,
        "text_1": [query],
        "text_2": [document],
    }

    return benchmark_data


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/score",
            workload_calculator=lambda data: len(data.get("text_1", [])),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=score_benchmark_generator,
                concurrency=10,
                runs=3
            )
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
