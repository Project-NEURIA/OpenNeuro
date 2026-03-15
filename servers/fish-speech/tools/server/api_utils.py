import asyncio
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Annotated, Any

import ormsgpack
from baize.datastructures import ContentType
from kui.asgi import (
    HTTPException,
    HttpRequest,
    JSONResponse,
    request,
)
from loguru import logger
from pydantic import BaseModel

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest
from tools.server.inference import inference_wrapper as inference

_STREAM_DONE = object()
_STREAM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-stream")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["tts"], default="tts")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=str,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-text-length", type=int, default=0)
    parser.add_argument("--listen", type=str, default="127.0.0.1:8080")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--load-4bit", action="store_true", help="Quantize entire pipeline to 4-bit (Slow AR, Fast AR, and DAC Decoder)")
    
    # Streaming / Batching Hyperparameters
    parser.add_argument("--max-seq-len", type=int, default=4096, help="Maximum KV cache length")
    parser.add_argument("--streaming-chunk-size", type=int, default=30, help="Number of new tokens per streaming chunk")
    parser.add_argument("--streaming-overlap-size", type=int, default=10, help="Number of overlap tokens for decoder context")

    return parser.parse_args()


class MsgPackRequest(HttpRequest):
    async def data(
        self,
    ) -> Annotated[
        Any,
        ContentType("application/msgpack"),
        ContentType("application/json"),
        ContentType("multipart/form-data"),
    ]:
        if self.content_type == "application/msgpack":
            return ormsgpack.unpackb(await self.body)

        elif self.content_type == "application/json":
            return await self.json

        elif self.content_type == "multipart/form-data":
            return await self.form

        raise HTTPException(
            HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            headers={
                "Accept": "application/msgpack, application/json, multipart/form-data"
            },
        )


async def inference_async(
    req: ServeTTSRequest,
    engine: TTSInferenceEngine,
):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[object] = asyncio.Queue()

    def producer():
        try:
            for chunk in inference(req, engine):
                if isinstance(chunk, bytes) and len(chunk) > 0:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

    _STREAM_EXECUTOR.submit(producer)

    while True:
        item = await queue.get()
        if item is _STREAM_DONE:
            break
        if isinstance(item, Exception):
            raise item
        # Yield chunks that are always multiples of 2 bytes (for int16)
        yield item


async def buffer_to_async_generator(buffer):
    yield buffer


def get_content_type(audio_format):
    if audio_format == "wav":
        return "audio/wav"
    elif audio_format == "flac":
        return "audio/flac"
    elif audio_format == "mp3":
        return "audio/mpeg"
    elif audio_format == "pcm":
        return "audio/pcm"
    else:
        return "application/octet-stream"


def wants_json(req):
    """Helper method to determine if the client wants a JSON response

    Parameters
    ----------
    req : Request
        The request object

    Returns
    -------
    bool
        True if the client wants a JSON response, False otherwise
    """
    q = req.query_params.get("format", "").strip().lower()
    if q in {"json", "application/json", "msgpack", "application/msgpack"}:
        return q == "json"
    accept = req.headers.get("Accept", "").strip().lower()
    return "application/json" in accept and "application/msgpack" not in accept


def format_response(response: BaseModel, status_code=200):
    """
    Helper function to format responses consistently based on client preference.

    Parameters
    ----------
    response : BaseModel
        The response object to format
    status_code : int
        HTTP status code (default: 200)

    Returns
    -------
    Response
        Formatted response in the client's preferred format
    """
    try:
        if wants_json(request):
            return JSONResponse(
                response.model_dump(mode="json"), status_code=status_code
            )

        return (
            ormsgpack.packb(
                response,
                option=ormsgpack.OPT_SERIALIZE_PYDANTIC,
            ),
            status_code,
            {"Content-Type": "application/msgpack"},
        )
    except Exception as e:
        logger.error(f"Error formatting response: {e}", exc_info=True)
        # Fallback to JSON response if formatting fails
        return JSONResponse(
            {"error": "Response formatting failed", "details": str(e)}, status_code=500
        )
