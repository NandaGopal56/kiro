from dotenv import load_dotenv
load_dotenv()

import asyncio
from stt_audio_processor import create_service

async def main():
    stt_audio_processor = create_service()
    await stt_audio_processor.start()
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
