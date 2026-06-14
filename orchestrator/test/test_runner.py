import pytest

from orchestrator.runner import Orchestrator


class FakeService:
    def __init__(self, name, events, fail_start=False):
        self.name = name
        self.events = events
        self.fail_start = fail_start

    async def start(self, **kwargs):
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(self.name)

    async def stop(self):
        self.events.append(f"stop:{self.name}")


@pytest.mark.asyncio
async def test_orchestrator_starts_and_stops_in_order():
    events = []
    orchestrator = Orchestrator([
        FakeService("a", events),
        FakeService("b", events),
    ])

    await orchestrator.start()
    await orchestrator.stop()

    assert events == ["start:a", "start:b", "stop:b", "stop:a"]


@pytest.mark.asyncio
async def test_orchestrator_cleans_up_started_services_on_failure():
    events = []
    orchestrator = Orchestrator([
        FakeService("a", events),
        FakeService("b", events, fail_start=True),
    ])

    with pytest.raises(RuntimeError):
        await orchestrator.start()

    assert events == ["start:a", "start:b", "stop:a"]
