import gc
import queue
from typing import Generator

import numpy as np
import torch
from loguru import logger

from fish_speech.inference_engine.reference_loader import ReferenceLoader
from fish_speech.inference_engine.utils import InferenceResult, wav_chunk_header
from fish_speech.inference_engine.vq_manager import VQManager
from fish_speech.models.dac.modded_dac import DAC
from fish_speech.models.text2semantic.inference import (
    GenerateRequest,
    GenerateResponse,
    WrappedGenerateResponse,
)
from fish_speech.utils import autocast_exclude_mps, set_seed
from fish_speech.utils.schema import ServeTTSRequest


class TTSInferenceEngine(ReferenceLoader, VQManager):

    def __init__(
        self,
        llama_queue: queue.Queue,
        decoder_model: DAC,
        precision: torch.dtype,
        compile: bool,
    ) -> None:

        super().__init__()

        self.llama_queue = llama_queue
        self.decoder_model = decoder_model
        self.precision = precision
        self.compile = compile

    @torch.inference_mode()
    def inference(self, req: ServeTTSRequest) -> Generator[InferenceResult, None, None]:
        """
        Main inference function:
        - Loads the reference audio and text.
        - Calls the LLAMA model for inference.
        - Decodes the VQ tokens to audio.
        """

        ref_id: str | None = req.reference_id
        prompt_tokens, prompt_texts = [], []
        # Load the reference audio and text based on id or hash
        if ref_id is not None:
            logger.info(f"Loading reference by id: {ref_id}")
            prompt_tokens, prompt_texts = self.load_by_id(ref_id, req.use_memory_cache)

        elif req.references:
            logger.info(f"Loading reference by hash: {len(req.references)} samples")
            prompt_tokens, prompt_texts = self.load_by_hash(
                req.references, req.use_memory_cache
            )
        else:
            logger.warning("No reference provided for inference")
        # Set the random seed if provided
        if req.seed is not None:
            set_seed(req.seed)
            logger.warning(f"set seed: {req.seed}")

        # Get the symbolic tokens from the LLAMA model
        response_queue = self.send_Llama_request(req, prompt_tokens, prompt_texts)

        # Get the sample rate from the decoder model
        if hasattr(self.decoder_model, "spec_transform"):
            sample_rate = self.decoder_model.spec_transform.sample_rate
        else:
            sample_rate = self.decoder_model.sample_rate

        # If streaming, send the header
        if req.streaming and req.format == "wav":
            yield InferenceResult(
                code="header",
                audio=(
                    sample_rate,
                    np.array(wav_chunk_header(sample_rate=sample_rate)),
                ),
                error=None,
            )

        segments = []
    
        while True:
            # Get the response from the LLAMA model
            wrapped_result: WrappedGenerateResponse = response_queue.get()
            if wrapped_result.status == "error":
                yield InferenceResult(
                    code="error",
                    audio=None,
                    error=(
                        wrapped_result.response
                        if isinstance(wrapped_result.response, Exception)
                        else Exception("Unknown error")
                    ),
                )
                break

            # Check the response type
            if not isinstance(wrapped_result.response, GenerateResponse):
                raise TypeError(
                    "Expected GenerateResponse, got {type(wrapped_result.response).__name__}"
                )

            result: GenerateResponse = wrapped_result.response
            if result.action != "next":
                segment = self.get_audio_segment(result)

                # Check for overlap/stride metadata in the text field
                if result.text and result.text.startswith("{") and result.text.endswith("}"):
                    import ast
                    try:
                        meta = ast.literal_eval(result.text)
                        overlap_tokens = meta.get("overlap", 0)
                        # Remove the audio samples corresponding to the overlap lookback
                        # Fish Speech is 50Hz, so each token is roughly 
                        # decoder_sample_rate / 50 samples.
                        # We calculate exact duration from segment length vs token count.
                        total_tokens = result.codes.shape[1]
                        if total_tokens > 0:
                            samples_per_token = segment.shape[0] / total_tokens
                            start_sample = int(overlap_tokens * samples_per_token)
                            
                            segment = segment[start_sample:]
                            
                            # Ensure the final segment length is even so total bytes are a multiple of 2 for int16
                            if segment.shape[0] % 2 != 0:
                                segment = segment[:-1]
                    except Exception as e:
                        logger.warning(f"Failed to parse streaming metadata: {e}")

                if req.streaming:  # Used only by the API server
                    yield InferenceResult(
                        code="segment",
                        audio=(sample_rate, segment),
                        error=None,
                    )
                segments.append(segment)
            else:
                break

        # Clean up the memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Edge case: no audio generated
        if len(segments) == 0:
            yield InferenceResult(
                code="error",
                audio=None,
                error=RuntimeError("No audio generated, please check the input text."),
            )
        else:
            # Streaming or not, return the final audio
            audio = np.concatenate(segments, axis=0)
            yield InferenceResult(
                code="final",
                audio=(sample_rate, audio),
                error=None,
            )

        return None

    def send_Llama_request(
        self, req: ServeTTSRequest, prompt_tokens: list, prompt_texts: list
    ) -> queue.Queue:
        """
        Send a request to the LLAMA model to generate the symbolic tokens.
        """
        # Get server-level args from app state via model manager
        # (This assumes engine.args exists, provided by initialize_app)
        max_seq_len = getattr(self, "max_seq_len", 4096)
        streaming_chunk_size = getattr(self, "streaming_chunk_size", 30)
        streaming_overlap_size = getattr(self, "streaming_overlap_size", 10)

        # Prepare the request
        request = dict(
            device=self.decoder_model.device,
            max_new_tokens=req.max_new_tokens,
            text=req.text,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            temperature=req.temperature,
            compile=self.compile,
            iterative_prompt=req.chunk_length > 0,
            chunk_length=req.chunk_length,
            prompt_tokens=prompt_tokens,
            prompt_text=prompt_texts,
            # Pass through streaming hyperparams
            max_seq_len=max_seq_len,
            streaming_chunk_size=streaming_chunk_size,
            streaming_overlap_size=streaming_overlap_size,
        )

        # Create a queue to get the response
        response_queue = queue.Queue()

        # Send the request to the LLAMA model
        self.llama_queue.put(
            GenerateRequest(
                request=request,
                response_queue=response_queue,
            )
        )

        return response_queue

    def get_audio_segment(self, result: GenerateResponse) -> np.ndarray:
        """
        Decode the VQ tokens to audio.
        """

        # Don't use autocast on MPS devices
        with autocast_exclude_mps(
            device_type=self.decoder_model.device.type, dtype=self.precision
        ):
            # Decode the symbolic tokens to audio
            segment = self.decode_vq_tokens(codes=result.codes)

        # Convert the audio to numpy
        return segment.float().cpu().numpy()
