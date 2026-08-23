"""The seam where a model plugs in - and the mock that lets us work without one.

Nothing here calls a model. This phase builds the socket, not the plug: a
protocol thin enough that any provider satisfies it, an adapter over Marker's
existing service layer so no new SDK enters the project, and a mock that makes
every downstream test runnable with no key, no network and no cost.

Provider independence is a hard requirement, so no provider name appears
anywhere except in the import string the operator passes on the command line.
Marker already ships adapters for Gemini, Claude, OpenAI, Azure, Vertex,
OpenRouter and Ollama; reusing them means this project adds no dependency and
gains every provider at once.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Type

from pydantic import BaseModel, Field

#: Which environment names can supply which service config key.
#:
#: Marker's own CLI performs one of these mappings (``GOOGLE_API_KEY`` ->
#: ``gemini_api_key``) inside its ``ConfigParser``. This package does not use
#: that parser, so the same wiring has to exist here - and while we are at it,
#: the other providers Marker ships are covered too, so adding one later is a
#: table entry rather than a code change.
#:
#: A key is only ever applied when the service class actually declares that
#: attribute, so no provider's own defaults are disturbed.
ENV_CONFIG_KEYS: Mapping[str, Sequence[str]] = {
    "gemini_api_key": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "claude_api_key": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
    "openai_api_key": ("OPENAI_API_KEY",),
    "openai_base_url": ("OPENAI_BASE_URL",),
    "azure_api_key": ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
    "azure_endpoint": ("AZURE_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
    "deployment_name": ("AZURE_DEPLOYMENT_NAME",),
    "openrouter_api_key": ("OPENROUTER_API_KEY",),
    "ollama_base_url": ("OLLAMA_BASE_URL", "OLLAMA_HOST"),
    "vertex_project_id": ("VERTEX_PROJECT_ID",),
}

#: Settings fields on Marker's own ``settings`` object worth consulting. This
#: is what makes ``local.env`` work: Marker reads that file into its settings,
#: and this is the only bridge from there to a service's config.
SETTINGS_CONFIG_KEYS: Mapping[str, str] = {
    "gemini_api_key": "GOOGLE_API_KEY",
}


def _marker_settings():
    """Marker's settings singleton, imported lazily.

    Lazy so that importing this module - which every test does - never pulls in
    Marker's import graph.
    """
    from marker.settings import settings  # noqa: PLC0415

    return settings


def build_service_config(
    service_cls: type,
    config: Optional[Dict[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    settings_obj: Any = None,
) -> Dict[str, Any]:
    """Fill in credentials a service needs but the caller did not pass.

    Precedence, strongest first:

    1. whatever the caller put in ``config`` - an explicit value always wins;
    2. Marker's settings, which is where ``local.env`` ends up;
    3. the process environment.

    Values are only added for attributes the service class declares, so this
    never injects a stray key into a provider that has no idea what it means.

    The returned dict carries secrets. It is handed straight to the service
    constructor and is never logged, printed, serialised, or stored on the
    client - see :class:`MarkerServiceClient`.
    """
    resolved: Dict[str, Any] = dict(config or {})
    environ = os.environ if env is None else env

    for attr, settings_field in SETTINGS_CONFIG_KEYS.items():
        if resolved.get(attr) or not hasattr(service_cls, attr):
            continue
        source = settings_obj if settings_obj is not None else _marker_settings()
        value = getattr(source, settings_field, None)
        if value:
            resolved[attr] = value

    for attr, names in ENV_CONFIG_KEYS.items():
        if resolved.get(attr) or not hasattr(service_cls, attr):
            continue
        for name in names:
            value = environ.get(name)
            if value:
                resolved[attr] = value
                break

    return resolved


class LLMRequest(BaseModel):
    """One structured request. Images are paths, resolved by the adapter."""

    prompt: str
    response_schema: Any
    image_paths: List[str] = Field(default_factory=list)
    max_retries: int = 2
    timeout: Optional[int] = None


class LLMClient(Protocol):
    """Everything the semantic layer needs from a model. Deliberately tiny."""

    def complete(self, request: LLMRequest) -> BaseModel:
        ...

    @property
    def model_id(self) -> str:
        ...


class MarkerServiceClient:
    """Adapter over ``marker.services.BaseService``.

    Marker's services already take a prompt, optional images and a pydantic
    response schema, and return a parsed object - the same contract this layer
    wants. Wrapping instead of reimplementing keeps provider support in one
    place and adds no dependency.

    The Marker import is local to the call so that importing this module - as
    every test does - never drags in Marker's import graph.
    """

    def __init__(self, service: Any, model_id: str = "unknown") -> None:
        self._service = service
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Explicit, so a stack trace or a debugger session can never print a
        # service object's credentials by way of this wrapper.
        return f"MarkerServiceClient(model_id={self._model_id!r})"

    @staticmethod
    def from_import_path(import_path: str, config: Optional[Dict] = None):
        """Build a client from a dotted path, e.g. a Marker service class.

        The provider is named only here, by the operator, and never inside the
        pipeline's logic.

        Credentials are resolved by :func:`build_service_config` and passed
        straight into the service constructor. The resulting dict is not kept:
        the key lives only inside the service object, and nothing in this
        package reads it back out.
        """
        from marker.util import strings_to_classes  # noqa: PLC0415

        service_cls = strings_to_classes([import_path])[0]
        service_config = build_service_config(service_cls, config)
        return MarkerServiceClient(
            service_cls(service_config), model_id=import_path
        )

    def complete(self, request: LLMRequest) -> BaseModel:
        images = _load_images(request.image_paths)
        return self._service(
            request.prompt,
            images or None,
            None,
            request.response_schema,
            max_retries=request.max_retries,
            timeout=request.timeout,
        )


def _load_images(paths: Sequence[str]) -> List[Any]:
    if not paths:
        return []
    from PIL import Image  # noqa: PLC0415

    return [Image.open(path) for path in paths]


class MockLLMClient:
    """A scripted stand-in, so the whole pipeline is testable without a model.

    Responses are queued by the test. Every request is recorded, which is how
    the tests assert the thing that matters most about this layer: that a model
    is only ever shown the evidence unit it is supposed to see.
    """

    def __init__(
        self,
        responses: Optional[Sequence[BaseModel]] = None,
        model_id: str = "mock",
    ) -> None:
        self._responses: List[BaseModel] = list(responses or [])
        self._model_id = model_id
        self.requests: List[LLMRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def queue(self, response: BaseModel) -> None:
        self._responses.append(response)

    def complete(self, request: LLMRequest) -> BaseModel:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                "MockLLMClient received a request with no queued response; "
                "the test must script every call it expects"
            )
        return self._responses.pop(0)


def response_schema_for(model: Type[BaseModel]) -> Type[BaseModel]:
    """Placeholder for the per-task schemas the semantic phase will add.

    Kept so the seam is visible: the semantic layer will pass a pydantic model
    describing exactly the fields it wants back, and a response that does not
    fit is rejected before it can reach the verifier.
    """
    return model
