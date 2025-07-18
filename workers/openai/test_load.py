from lib.test_utils import test_load_cmd, test_args
from .data_types.server import CompletionsData
import os

WORKER_ENDPOINT = "/v1/completions"

if __name__ == "__main__":
    # Check if MODEL_NAME environment variable is set
    model_name_set = os.environ.get("MODEL_NAME") is not None

    # Add model argument - required only if MODEL_NAME is not set
    test_args.add_argument(
        "--model",
        dest="model",
        required=not model_name_set,
        help="Model to use for completions request (required if MODEL_NAME env var not set)",
    )

    # Parse known args to get model early, before test_load_cmd adds its args
    known_args, _ = test_args.parse_known_args()

    # Set environment variable if model was provided
    if hasattr(known_args, "model") and known_args.model:
        os.environ["MODEL_NAME"] = known_args.model
        print(f"Set MODEL_NAME environment variable to: {known_args.model}")

    # Now call test_load_cmd normally - it will add its own args and re-parse
    test_load_cmd(CompletionsData, WORKER_ENDPOINT, arg_parser=test_args)
