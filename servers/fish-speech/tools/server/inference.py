from http import HTTPStatus

import numpy as np
from kui.asgi import HTTPException

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.utils.schema import ServeTTSRequest

AMPLITUDE = 32768  # Needs an explaination


def inference_wrapper(req: ServeTTSRequest, engine: TTSInferenceEngine):
    """
    Wrapper for the inference function.
    Used in the API server.
    """
    count = 0
    for result in engine.inference(req):
        match result.code:
            case "header":
                if isinstance(result.audio, tuple) and req.format == "wav":
                    yield result.audio[1]

            case "error":
                raise HTTPException(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    content=str(result.error),
                )

            case "segment":
                count += 1
                if isinstance(result.audio, tuple):
                    audio_data = result.audio[1]
                    yield (audio_data * AMPLITUDE).astype(np.int16).tobytes()

            case "final":
                if req.streaming:
                    return None

                count += 1
                if isinstance(result.audio, tuple):
                    audio_data = result.audio[1]
                    yield (audio_data * AMPLITUDE).astype(np.int16).tobytes()
                return None  # Stop the generator

    if count == 0:
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            content="No audio generated, please check the input text.",
        )
