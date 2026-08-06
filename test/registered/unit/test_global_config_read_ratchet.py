"""Ratchet guard: process-global config reads may only decrease.

``get_server_args()`` returns the published ``ServerArgs`` — one process's
startup record. Config decisions read the namespace accessors instead
(``get_exec()`` / ``get_memory()`` / …), which carry the resolved value
including post-publish overrides, and per-runner values come from the runner
that owns them.

Two shapes count as a read: the direct ``get_server_args().field``, and the
alias ``sa = get_server_args()`` followed by ``sa.field`` in the same function.
A whole-object pass (``def f(server_args)``) is not a global read and is not
counted — there the caller decided which instance to hand over.

Business code no longer reads the published record for a config value at all:
the baselines are zero for both shapes, over the whole package minus the two
modules that own the slot.

Where the remaining reads live (``runtime_context.py``, exempt by module):

- **Derived members.** ``@property`` / method members of ``ServerArgs``
  (``mamba_cache_chunk_size``, ``max_speculative_num_draft_tokens``,
  ``use_mla_backend()``, ``get_attention_backends()``, ``get_model_config()``,
  ``cutedsl_moe_max_num_tokens()``) are computed from several fields plus the HF
  config, so they are not namespace leaves and ``ServerArgs`` is their only
  home. ``runtime_context`` exposes each one as a named accessor
  (``mamba_cache_chunk_size()`` …) and is the only module that reads the slot
  for them.
- **Config-intent reads of live-shadowed sizes.** ``get_parallel()`` shadows
  ``tp/pp/dcp/attn_cp/moe_dp_size`` with the live topology, and a few call sites
  need what was *configured*: the ``configured_*_size()`` accessors. Their
  reasons, per call site:

  - ``dsa_indexer`` gates ``pp_size > 1 and not get_pp_group()...``, and the
    short circuit is the point: with PP off the group is never touched, which is
    what lets the ``Indexer`` be constructed before distributed init.
  - ``allocation`` asks whether DCP was *configured*; the live property reads
    ``get_dcp_group()``, installed only when DCP is on.
  - ``cuda_ipc_transport_utils`` runs in the tokenizer process, which has no
    groups at all.
  - ``dp_attention``'s ``attn_cp_size > moe_dp_size``: the configuration the
    predicate detects is the one where ``initialize_model_parallel`` aliases
    ``_MOE_DP`` to ``_ATTN_CP``, so the live sizes are equal there and a live
    comparison is always false.
  - ``model_loader/loader`` reports both: the same dict carries the live
    ``moe_dp_size`` under ``"dp"``, so this entry is the configured intent.

A whole-object pass (``def f(server_args)``) is not a global read and is not
counted -- there the caller decided which instance to hand over. An optional
parameter that falls back to the global (``f(server_args=None)``) hides one,
so those fallbacks were removed; the ratchet cannot see them and the census
tool in the context repo is what audits that shape.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import unittest
from pathlib import Path

import sglang
from sglang.test.test_utils import CustomTestCase

# srt is the migrated surface; the rest of the package has no reads today and is
# scanned so a new one cannot appear there unnoticed.
_PACKAGE_ROOT = Path(next(iter(sglang.__path__)))

# The modules that own the slot: runtime_context publishes it and exposes the
# named accessors for the derived members, server_args/arg_groups ARE the
# resolution pipeline.
_SLOT_OWNERS = ("srt/runtime_context.py", "srt/server_args.py", "srt/arg_groups/")


_DIRECT_BASELINE = 0
_ALIAS_BASELINE = 0


def _is_global_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "get_server_args"
    )


def _collect(rel: str, tree: ast.AST):
    """The (direct, alias) field reads in one module."""
    direct, alias = [], []

    def counted(attr: str) -> bool:
        return True

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and _is_global_call(node.value)
            and counted(node.attr)
        ):
            direct.append(f"{rel}:{node.lineno}: get_server_args().{node.attr}")

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in list(node.args.args) + list(node.args.kwonlyargs)}
        bound = {}
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign) and _is_global_call(inner.value):
                for target in inner.targets:
                    if isinstance(target, ast.Name) and target.id not in params:
                        bound.setdefault(target.id, inner.lineno)
        if not bound:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id in bound
                and inner.lineno >= bound[inner.value.id]
                and counted(inner.attr)
            ):
                alias.append(
                    f"{rel}:{inner.lineno}: {inner.value.id}.{inner.attr} "
                    f"(bound from get_server_args() at line {bound[inner.value.id]})"
                )
    return direct, alias


def _field_reads():
    direct, alias = [], []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(_PACKAGE_ROOT).as_posix()
        if rel.startswith(_SLOT_OWNERS):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        module_direct, module_alias = _collect(rel, tree)
        direct += module_direct
        alias += module_alias
    return direct, alias


class TestGlobalConfigReadRatchet(CustomTestCase):
    def _check(self, kind, reads, baseline):
        if len(reads) > baseline:
            self.fail(
                f"{kind} process-global config field reads grew: {len(reads)} > "
                f"baseline {baseline}. Read the namespace accessor for the "
                "field's namespace, or the owning runner for a per-runner "
                "field:\n" + "\n".join(reads)
            )
        if len(reads) < baseline:
            self.fail(
                f"{kind} process-global config field reads shrank: {len(reads)} < "
                f"baseline {baseline}. Lower the baseline in this file to lock "
                "in the progress."
            )

    def test_global_field_reads_match_the_baseline(self):
        direct, alias = _field_reads()
        self._check("direct", direct, _DIRECT_BASELINE)
        self._check("alias-form", alias, _ALIAS_BASELINE)


if __name__ == "__main__":
    unittest.main()
