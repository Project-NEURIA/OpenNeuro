/**
 * AudioWorklet processor for low-latency microphone capture.
 *
 * Buffers 256 samples (16ms at 16 kHz) before posting an Int16 PCM
 * chunk to the main thread, replacing the deprecated ScriptProcessorNode
 * which ran on the main thread with 1024-sample (64ms) buffers.
 */

const processorCode = `
class AudioInputProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(256);
    this._pos = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    for (let i = 0; i < input.length; i++) {
      this._buf[this._pos++] = input[i];
      if (this._pos >= 256) {
        const int16 = new Int16Array(256);
        for (let j = 0; j < 256; j++) {
          const s = Math.max(-1, Math.min(1, this._buf[j]));
          int16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._pos = 0;
      }
    }
    return true;
  }
}

registerProcessor('audio-input-processor', AudioInputProcessor);
`;

let _blobUrl: string | null = null;

export function getAudioInputProcessorUrl(): string {
  if (!_blobUrl) {
    const blob = new Blob([processorCode], { type: "application/javascript" });
    _blobUrl = URL.createObjectURL(blob);
  }
  return _blobUrl;
}
