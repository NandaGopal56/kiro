"""Live interaction web UI package."""


def create_service(*args, **kwargs):
    from .ui_service import create_service as _create_service

    return _create_service(*args, **kwargs)


__all__ = ["create_service"]
