from dotenv import load_dotenv
load_dotenv()

import asyncio
from stt_audio_processor.stt_processor import STTAudioProcessorService

stt_audio_processor = STTAudioProcessorService()

async def main():
    await stt_audio_processor.start()


if __name__ == "__main__":
    asyncio.run(main())