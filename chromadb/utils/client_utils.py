from typing import Any, Optional


def get_client_options(timeout: Optional[float]) -> dict[str, Any]:
    client_options = {}
    if timeout:
        client_options["timeout"] = timeout
    return client_options
