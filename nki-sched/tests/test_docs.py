"""The docs are checked against the code, so they cannot silently fall behind it."""

import inspect
import os

import pytest

from nki_sched import Sched
from nki_sched.instrs import INSTRS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


@pytest.mark.parametrize("instr", sorted(INSTRS))
def test_every_instruction_is_documented(instr):
    assert f"`{instr}`" in read("docs/ATTENTION.md") + read("SYNTAX.md")


PUBLIC = sorted(n for n, f in inspect.getmembers(Sched, inspect.isfunction) if not n.startswith("_"))


@pytest.mark.parametrize("method", PUBLIC)
def test_every_public_primitive_is_documented_in_the_syntax_reference(method):
    assert f"`{method}(" in read("SYNTAX.md") or f"s.{method}(" in read("SYNTAX.md") or f"`{method}`" in read("SYNTAX.md")


@pytest.mark.parametrize("method", [m for m in PUBLIC if m not in ("show", "source", "loops")])
def test_every_scheduling_primitive_is_in_the_agent_description(method):
    """AGENT.md is the language as shown to a kernel agent: it must list every primitive the agent may call"""
    assert f"s.{method}(" in read("AGENT.md") or f"`{method}" in read("AGENT.md")
