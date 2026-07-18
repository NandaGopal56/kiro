from __future__ import annotations

import argparse

from dotenv import find_dotenv, load_dotenv

from shared.logging import get_logger
from vision.vlm.client import VLMClient

logger = get_logger("vision.vlm.__main__", log_file="vision_vlm.log")
load_dotenv(find_dotenv())


def _camera_caption(client: VLMClient, prompt: str, every_n: int, max_frames: int = 0) -> None:
    import cv2

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera.")

    count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            count += 1
            if count % every_n == 0:
                resp = client.analyze(frame, prompt)
                print(f"[VLM] {resp.text}")
            cv2.imshow("VLM", frame)
            if max_frames and count >= max_frames:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VLM standalone.")
    parser.add_argument("--backend", default="openai")
    parser.add_argument("--prompt", default="Describe what is happening in this scene.")
    parser.add_argument("--every-n", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    client = VLMClient(backend=args.backend)
    logger.info("Starting VLM client (backend=%s)", args.backend)
    _camera_caption(client, args.prompt, args.every_n, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
