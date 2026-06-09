from agents import bot


def test_workflow_is_lazy_on_import():
    assert bot._workflow is None
