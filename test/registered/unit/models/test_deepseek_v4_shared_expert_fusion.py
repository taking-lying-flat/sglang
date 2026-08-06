import unittest
from types import SimpleNamespace

from sglang.srt.layers.moe.utils import is_shared_experts_fusion_disabled
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.runtime_context import get_context, get_exec, get_flags
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDeepseekV4SharedExpertFusionPolicy(unittest.TestCase):
    """The gate records its decision on the ACTIVE moe flag (per build; a
    draft's gate writes for the draft's layers only) — the config bag keeps
    the user's intent untouched."""

    def _make_model(self, n_shared_experts=1):
        return SimpleNamespace(
            config=SimpleNamespace(n_shared_experts=n_shared_experts)
        )

    def _publish(self, enforce):
        override = get_context().override_server_args(
            enforce_shared_experts_fusion=enforce
        )
        override.install()
        self.addCleanup(override.restore)
        get_flags().moe.disable_shared_experts_fusion = None
        self.addCleanup(
            lambda: setattr(get_flags().moe, "disable_shared_experts_fusion", None)
        )

    def test_disables_shared_fusion_without_enforce(self):
        self._publish(enforce=False)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 0)
        # The decision lands on the ACTIVE flag; the config intent is untouched.
        self.assertTrue(is_shared_experts_fusion_disabled())
        self.assertFalse(get_exec().moe.disable_shared_experts_fusion)

    def test_enables_shared_fusion_when_enforced(self):
        self._publish(enforce=True)
        model = self._make_model()

        DeepseekV4ForCausalLM.determine_num_fused_shared_experts(model)

        self.assertEqual(model.num_fused_shared_experts, 1)
        self.assertFalse(is_shared_experts_fusion_disabled())


if __name__ == "__main__":
    unittest.main()
