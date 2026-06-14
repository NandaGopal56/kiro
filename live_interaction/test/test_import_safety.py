def test_live_interaction_package_import_is_lazy():
    import live_interaction

    assert hasattr(live_interaction, "create_service")


def test_create_service_can_inject_dependencies():
    import pytest

    pytest.importorskip("fastapi")

    from live_interaction.ui_service import create_service

    async def fake_agent(message: str, thread_id: int = 1):
        yield f"echo:{message}:{thread_id}"

    service = create_service(bus=object(), agent_client=fake_agent, port=9000)

    assert service.port == 9000
    assert service.agent_client is fake_agent
