"""Resolution is a pure function of the raw input plus this node's environment.

The end state for the configuration tier keeps ``ServerArgs`` at the user's raw
input and lets every process that publishes derive the resolved values itself
(bags do not cross a process boundary — a child projects its own from the record
it is handed). That is only sound if resolving the same raw input twice gives
the same answer, so this pins it:

- twice in this process, from equal raw inputs, every field agrees;
- the resolution is not order-dependent on a shared registry (a second config
  resolved after the first does not inherit its declarations);
- the raw record the two started from is itself unchanged by resolving a
  sibling.

A failure here means some resolution step reads state it also writes, and the
"re-derive in the child" contract would silently diverge between the launcher
and its schedulers.
"""

import dataclasses
import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# Fields whose value is a fresh object per construction (identity differs, and
# equality is not defined for all of them) or a deliberately random seed.
_NOT_COMPARABLE = frozenset({"random_seed"})


class TestResolutionIsReproducible(CustomTestCase):
    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="resolution_repro_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return config_dir

    def _resolved(self, model_path: str, **kwargs) -> ServerArgs:
        # device="cuda" keeps the golden path host-independent: an
        # accelerator-less runner resolves only the base platform, where
        # get_device() raises.
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("random_seed", 42)
        return ServerArgs(model_path=model_path, **kwargs)

    def _comparable(self, server_args: ServerArgs) -> dict:
        out = {}
        for field in dataclasses.fields(server_args):
            if field.name in _NOT_COMPARABLE:
                continue
            value = getattr(server_args, field.name)
            # Nested dataclasses (cuda_graph_config) compare structurally.
            out[field.name] = (
                dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
            )
        return out

    def test_two_resolutions_of_the_same_input_agree(self):
        model_path = self._config_dir()
        first = self._resolved(model_path)
        second = self._resolved(model_path)
        self.assertEqual(self._comparable(first), self._comparable(second))

    def test_a_resolution_does_not_leak_into_the_next(self):
        # The declaration registry is process-global; a config resolved with an
        # explicit backend must not shift the default the next one picks.
        model_path = self._config_dir()
        explicit = self._resolved(model_path, attention_backend="triton")
        self.assertEqual(explicit.attention_backend, "triton")
        default_after = self._resolved(model_path)
        default_alone = self._resolved(self._config_dir())
        self.assertEqual(
            default_after.attention_backend, default_alone.attention_backend
        )

    def test_resolving_a_sibling_leaves_the_first_alone(self):
        model_path = self._config_dir()
        first = self._resolved(model_path)
        snapshot = self._comparable(first)
        self._resolved(model_path, tp_size=2, chunked_prefill_size=1024)
        self.assertEqual(self._comparable(first), snapshot)

    def test_the_declaration_provenance_is_reproducible(self):
        model_path = self._config_dir()
        first = self._resolved(model_path)
        second = self._resolved(model_path)
        self.assertEqual(
            getattr(first, "_resolved_overrides", None),
            getattr(second, "_resolved_overrides", None),
        )


if __name__ == "__main__":
    unittest.main()
