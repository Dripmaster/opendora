from __future__ import annotations

import asyncio
import signal

from orchestrator.app import OrchestratorApp


async def _main() -> None:
    app = OrchestratorApp()
    stop_event = asyncio.Event()

    async def _shutdown(sig_name: str) -> None:
        try:
            await app.stop()
        finally:
            stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(_shutdown(s.name)))

    await app.start()
    await stop_event.wait()


def run() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
