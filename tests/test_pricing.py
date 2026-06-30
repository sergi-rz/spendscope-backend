from app.services import pricing
from app.services.oplog import anon_user


def test_cost_for_known_openai_model():
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out.
    cost = pricing.cost_usd("gpt-4o-mini", 1_000_000, 1_000_000)
    assert round(cost, 4) == round(0.15 + 0.60, 4)


def test_cost_is_zero_for_self_hosted_models():
    # gemma4/qwen run on nan.builders (no per-token price) → 0.
    assert pricing.cost_usd("gemma4", 5000, 5000) == 0.0
    assert pricing.cost_usd(None, 5000, 5000) == 0.0


def test_cost_case_insensitive():
    assert pricing.cost_usd("GPT-4o-Mini", 1_000_000, 0) == pricing.cost_usd("gpt-4o-mini", 1_000_000, 0)


def test_anon_user_is_stable_and_irreversible():
    a, b = anon_user("device-abc"), anon_user("device-abc")
    assert a == b and a != "device-abc" and len(a) == 32
    assert anon_user("device-abc") != anon_user("device-xyz")
    assert anon_user(None) is None
    assert anon_user("") is None
