"""A prepared hosted service must not re-enter its profile scope on every watcher tick.

``run_startup.py::_hosted_room_worker_watcher`` calls ``ensure_hosted_service`` once per second for
the life of the messaging gateway, per serving authority. Every entry used to wrap the whole body
in ``owner_scope(authority)`` — which rebuilds that profile's secret map (``.env`` read, external
sources, managed env) and its terminal policy (managed-dir stat, ``.env``, ``config.yaml``) — even
when the service already existed with its transport installed. That is the frame three of the
fourteen blocked-loop dumps in PRX-5524 sampled
(``_hosted_room_worker_watcher`` -> ``ensure_hosted_service`` -> ``owner_scope`` ->
``_profile_runtime_scope`` -> ``build_profile_secret_scope``/``build_profile_terminal_scope``).

Contract: the prepared case is two getattrs and no scope; the unprepared case still enters the
scope it needs to build and install the transport.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from gateway import session_hosted_service as hosted


class _ScopeSpy:
    def __init__(self):
        self.entries = []

    def __call__(self, authority, **kwargs):
        self.entries.append(authority)
        return contextlib.nullcontext()


@pytest.fixture
def scope_spy(monkeypatch):
    spy = _ScopeSpy()
    monkeypatch.setattr(hosted, 'owner_scope', spy)
    return spy


@pytest.mark.asyncio
async def test_a_prepared_service_tick_is_two_getattrs_and_no_profile_scope(scope_spy, tmp_path):
    service = SimpleNamespace(_transport_installed=True)
    authority = SimpleNamespace(profile_id=str(tmp_path), hosted_room_service=service)
    runner = SimpleNamespace(session_control_server=None)

    await hosted._ensure_hosted_service(runner, authority)

    assert scope_spy.entries == []
    assert authority.hosted_room_service is service


@pytest.mark.asyncio
async def test_a_service_without_its_transport_still_enters_the_scope_and_installs(
        scope_spy, monkeypatch, tmp_path):
    installed = []

    def install_transport(server, authority, loop, *, attest):
        installed.append((server, authority, attest))

    monkeypatch.setattr('gateway.session_hosted_transport.install_hosted_transport', install_transport)
    service = SimpleNamespace(_transport_installed=False, attest=lambda *a: True)
    authority = SimpleNamespace(profile_id=str(tmp_path), hosted_room_service=service)
    server = object()
    runner = SimpleNamespace(session_control_server=server)

    await hosted._ensure_hosted_service(runner, authority)

    assert scope_spy.entries == [authority]
    assert service._transport_installed is True
    assert installed == [(server, authority, service.attest)]


@pytest.mark.asyncio
async def test_a_missing_service_is_built_off_loop_inside_the_scope(scope_spy, monkeypatch, tmp_path):
    """The cold path keeps its scope (the constructor hydrates that profile's store) and still
    builds the service on a worker thread."""
    import threading

    seen = {}

    class _Built:
        _transport_installed = True

        def __init__(self, authority, loop):
            seen['thread'] = threading.get_ident()
            seen['authority'] = authority

    monkeypatch.setattr(hosted, 'CanonicalHostedRoomService', _Built)
    monkeypatch.setattr('gateway.session_hosted_transport.install_hosted_transport',
                        lambda *a, **kw: None)
    authority = SimpleNamespace(profile_id=str(tmp_path), hosted_room_service=None)
    loop_thread = threading.get_ident()

    await hosted._ensure_hosted_service(SimpleNamespace(session_control_server=None), authority)

    assert scope_spy.entries == [authority]
    assert seen['thread'] != loop_thread
    assert seen['authority'] is authority
    assert isinstance(authority.hosted_room_service, _Built)
