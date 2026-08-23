"""Regression tests for credential wiring.

Marker reads ``local.env`` into its own settings and its CLI turns
``GOOGLE_API_KEY`` into ``gemini_api_key`` when it builds a service. This
package does not use that CLI, so without the same bridge a correctly placed
key never reaches the provider and the run fails with an assertion that looks
nothing like "missing key".

These tests check the bridge and nothing else: no provider is constructed, no
network is touched, and no real credential is used anywhere - every value here
is an obvious dummy.
"""

from __future__ import annotations

import unittest

from content_assistant.structuring.semantic.llm import (
    ENV_CONFIG_KEYS,
    SETTINGS_CONFIG_KEYS,
    MarkerServiceClient,
    build_service_config,
)

DUMMY = "dummy-not-a-real-key"


class FakeGeminiService:
    """Stands in for GoogleGeminiService: declares the same config attribute."""

    gemini_api_key = None
    gemini_model_name = "x"


class FakeClaudeService:
    claude_api_key = None


class FakeSettings:
    def __init__(self, google_api_key=None):
        self.GOOGLE_API_KEY = google_api_key


class SettingsBridgeTests(unittest.TestCase):
    """The path that makes local.env work."""

    def test_google_api_key_becomes_gemini_api_key(self):
        config = build_service_config(
            FakeGeminiService,
            config=None,
            env={},
            settings_obj=FakeSettings(google_api_key=DUMMY),
        )
        self.assertEqual(config["gemini_api_key"], DUMMY)

    def test_an_empty_settings_value_adds_nothing(self):
        config = build_service_config(
            FakeGeminiService, config=None, env={}, settings_obj=FakeSettings("")
        )
        self.assertNotIn("gemini_api_key", config)

    def test_an_explicit_config_value_always_wins(self):
        config = build_service_config(
            FakeGeminiService,
            config={"gemini_api_key": "explicit-wins"},
            env={"GOOGLE_API_KEY": "from-env"},
            settings_obj=FakeSettings("from-settings"),
        )
        self.assertEqual(config["gemini_api_key"], "explicit-wins")

    def test_settings_outrank_the_process_environment(self):
        config = build_service_config(
            FakeGeminiService,
            config=None,
            env={"GOOGLE_API_KEY": "from-env"},
            settings_obj=FakeSettings("from-settings"),
        )
        self.assertEqual(config["gemini_api_key"], "from-settings")


class EnvironmentFallbackTests(unittest.TestCase):
    def test_environment_supplies_the_key_when_settings_do_not(self):
        config = build_service_config(
            FakeGeminiService,
            config=None,
            env={"GOOGLE_API_KEY": DUMMY},
            settings_obj=FakeSettings(None),
        )
        self.assertEqual(config["gemini_api_key"], DUMMY)

    def test_an_alternative_env_name_is_accepted(self):
        config = build_service_config(
            FakeGeminiService,
            config=None,
            env={"GEMINI_API_KEY": DUMMY},
            settings_obj=FakeSettings(None),
        )
        self.assertEqual(config["gemini_api_key"], DUMMY)

    def test_the_first_listed_name_takes_priority(self):
        config = build_service_config(
            FakeGeminiService,
            config=None,
            env={"GOOGLE_API_KEY": "first", "GEMINI_API_KEY": "second"},
            settings_obj=FakeSettings(None),
        )
        self.assertEqual(config["gemini_api_key"], "first")


class ProviderIsolationTests(unittest.TestCase):
    """Nothing is injected into a provider that does not ask for it."""

    def test_a_gemini_key_never_reaches_a_claude_service(self):
        config = build_service_config(
            FakeClaudeService,
            config=None,
            env={"GOOGLE_API_KEY": DUMMY},
            settings_obj=FakeSettings(DUMMY),
        )
        self.assertNotIn("gemini_api_key", config)

    def test_each_provider_gets_only_its_own_key(self):
        config = build_service_config(
            FakeClaudeService,
            config=None,
            env={"ANTHROPIC_API_KEY": DUMMY, "OPENAI_API_KEY": "other"},
            settings_obj=FakeSettings(None),
        )
        self.assertEqual(config, {"claude_api_key": DUMMY})

    def test_a_service_declaring_nothing_gets_an_untouched_config(self):
        class Bare:
            pass

        config = build_service_config(
            Bare,
            config={"mode": "x"},
            env={"GOOGLE_API_KEY": DUMMY, "ANTHROPIC_API_KEY": DUMMY},
            settings_obj=FakeSettings(DUMMY),
        )
        self.assertEqual(config, {"mode": "x"})

    def test_unrelated_config_values_are_preserved(self):
        config = build_service_config(
            FakeGeminiService,
            config={"gemini_model_name": "some-model"},
            env={},
            settings_obj=FakeSettings(DUMMY),
        )
        self.assertEqual(config["gemini_model_name"], "some-model")
        self.assertEqual(config["gemini_api_key"], DUMMY)

    def test_the_caller_dict_is_not_mutated(self):
        original = {"gemini_model_name": "m"}
        build_service_config(
            FakeGeminiService,
            config=original,
            env={},
            settings_obj=FakeSettings(DUMMY),
        )
        self.assertEqual(original, {"gemini_model_name": "m"})


class MappingTableTests(unittest.TestCase):
    def test_every_provider_marker_ships_has_an_entry(self):
        for attr in (
            "gemini_api_key",
            "claude_api_key",
            "openai_api_key",
            "azure_api_key",
            "openrouter_api_key",
            "ollama_base_url",
        ):
            self.assertIn(attr, ENV_CONFIG_KEYS)

    def test_the_settings_bridge_covers_the_key_marker_itself_reads(self):
        self.assertEqual(SETTINGS_CONFIG_KEYS["gemini_api_key"], "GOOGLE_API_KEY")


class SecrecyTests(unittest.TestCase):
    """A key must not be recoverable from anything this package exposes."""

    def test_the_client_repr_shows_no_credential(self):
        class Service:
            def __init__(self):
                self.gemini_api_key = DUMMY

        client = MarkerServiceClient(Service(), model_id="provider.path")
        text = repr(client)
        self.assertNotIn(DUMMY, text)
        self.assertIn("provider.path", text)

    def test_the_client_reports_only_the_import_path(self):
        client = MarkerServiceClient(object(), model_id="marker.services.x.Y")
        self.assertEqual(client.model_id, "marker.services.x.Y")
        self.assertNotIn("key", client.model_id.lower())


if __name__ == "__main__":
    unittest.main()
