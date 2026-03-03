from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from typing import NamedTuple, cast

import discord
from discord.sinks import PCMSink, Sink
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import Component
from src.core.frames import AudioDataFormat, AudioFrame, InterruptFrame

# Global Discord bot instance and event loop
_discord_bot: discord.Bot | None = None
_discord_loop: asyncio.AbstractEventLoop | None = None
_discord_thread: threading.Thread | None = None
_discord_running = False
_discord_lock = threading.Lock()
_active_discord_io: DiscordIO | None = None

# Global voice state for persistence across graph restarts
_voice_clients: dict[int, discord.VoiceClient] = {}
_rings: dict[int, deque[bytes]] = {}
_buffer: dict[int, deque[bytes]] = {}
_playback_tasks: dict[int, asyncio.Task[None]] = {}


class DiscordConfig(BaseModel):
    token_env_var: str = "DISCORD_TOKEN"
    sample_rate: int = 48000
    channels: int = 2
    audio_buffer_seconds: int = 64
    guild_ids: list[int] = []


class DiscordInputs(NamedTuple):
    audio: Receiver[AudioFrame]
    interrupt: Receiver[InterruptFrame] | None = None


class DiscordOutputs(NamedTuple):
    audio: Sender[AudioFrame]


class DiscordIO(Component[DiscordInputs, DiscordOutputs]):
    """Discord audio conduit that handles both input and output."""

    def __init__(self, config: DiscordConfig) -> None:
        super().__init__()
        self.config: DiscordConfig = config

        token = os.getenv(self.config.token_env_var)
        if not token:
            raise ValueError(f"Environment variable {self.config.token_env_var} must be set")
        self.token = token

        self.max_frames = self.config.audio_buffer_seconds * 50  # 20ms frames
        # Placeholder sender until run() wires the real one
        self._output_audio: Sender[AudioFrame] = Sender()

        print(f"[DiscordIO] DiscordIO initialized, guild_ids: {self.config.guild_ids}")

        self._ensure_discord_running()

    def _ensure_discord_running(self) -> None:
        global _discord_bot, _discord_loop, _discord_thread, _discord_running
        with _discord_lock:
            if _discord_running:
                return

            def run_discord() -> None:
                global _discord_bot, _discord_loop, _discord_running
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _discord_loop = loop

                intents = discord.Intents.default()
                bot = discord.Bot(intents=intents)
                _discord_bot = bot
                self._register_handlers_for_bot(bot)

                _discord_running = True
                try:
                    loop.run_until_complete(bot.start(self.token))
                except Exception as e:
                    print(f"[DiscordIO] Bot error: {e}")
                finally:
                    _discord_running = False

            _discord_thread = threading.Thread(target=run_discord, daemon=True)
            _discord_thread.start()

            # Wait for bot
            while _discord_bot is None:
                threading.Event().wait(0.1)

    def _register_handlers_for_bot(self, bot: discord.Bot) -> None:
        @bot.event
        async def on_ready() -> None:
            print(f"[DiscordIO] Bot ready: {bot.user}")

        @bot.slash_command(name="join", guild_ids=self.config.guild_ids or None)
        async def join(ctx: discord.ApplicationContext) -> None:
            member = ctx.author
            if not isinstance(member, discord.Member) or member.voice is None:
                await ctx.respond("Join a voice channel first")
                return

            if ctx.guild is None:
                await ctx.respond("Must be used in a guild")
                return

            gid = ctx.guild.id
            ring: deque[bytes] = deque(maxlen=2000) # Use a sane default or active's
            if _active_discord_io:
                ring = deque(maxlen=_active_discord_io.max_frames)
            _rings[gid] = ring

            channel = member.voice.channel
            if channel is None:
                await ctx.respond("Join a voice channel first")
                return

            vc = cast(discord.VoiceClient, await channel.connect())
            sink = _DiscordSink()

            vc.start_recording(sink, lambda *_args: None)

            _voice_clients[gid] = vc
            _buffer[gid] = deque(maxlen=1000)
            if _discord_loop:
                task = asyncio.run_coroutine_threadsafe(
                    self._playback_loop(gid), _discord_loop
                )
                _playback_tasks[gid] = cast(asyncio.Task[None], task)
            await ctx.respond("Connected")

        @bot.slash_command(name="leave", guild_ids=self.config.guild_ids or None)
        async def leave(ctx: discord.ApplicationContext) -> None:
            if ctx.guild is None:
                await ctx.respond("Must be used in a guild")
                return

            gid = ctx.guild.id
            vc = ctx.guild.voice_client
            if vc:
                await vc.disconnect()
                _rings.pop(gid, None)
                _voice_clients.pop(gid, None)
                _buffer.pop(gid, None)
                task = _playback_tasks.pop(gid, None)
                if task:
                    task.cancel()
            await ctx.respond("Disconnected")

    async def _playback_loop(self, guild_id: int) -> None:
        vc = _voice_clients[guild_id]
        buffer = _buffer[guild_id]
        source = _DiscordAudioSource(buffer)
        vc.play(source)
        while vc.is_connected():
            if not _discord_running:
                break
            await asyncio.sleep(1.0)

    def run(self, inputs: DiscordInputs, outputs: DiscordOutputs) -> None:
        global _active_discord_io
        _active_discord_io = self
        self._output_audio = outputs.audio

        print("[DiscordIO] Starting Discord processing")

        if inputs.interrupt is not None:
            interrupt_recv = inputs.interrupt

            def handle_interrupts() -> None:
                for frame in interrupt_recv(self):
                    if frame is None:
                        break
                    for b in _buffer.values():
                        b.clear()

            threading.Thread(target=handle_interrupts, daemon=True).start()

        for frame in inputs.audio(self):
            if frame is None:
                break

            # Use AudioFrame.get for resampling/reformatting
            pcm_data = frame.get(
                sample_rate=self.config.sample_rate,
                num_channels=self.config.channels,
                data_format=AudioDataFormat.PCM16,
            )

            for b in _buffer.values():
                b.append(pcm_data)

        print("[DiscordIO] Discord processing stopped")


class _DiscordSink(PCMSink):
    def write(self, data: bytes, user: object) -> None:
        if _active_discord_io:
            gid = None
            # Find the guild id for this sink if needed, but we can just use ring
            # Actually, _DiscordSink doesn't know its guild unless we tell it.
            # But the ring is global anyway. 
            # Wait, the ring should be per-guild.
            # Let's simplify and just send to the active IO.
            _active_discord_io._output_audio.send(
                AudioFrame.new(
                    data=data,
                    sample_rate=48000, # Discord is always 48k
                    channels=2, # Discord is always stereo
                )
            )


class _DiscordAudioSource(discord.AudioSource):
    def __init__(self, buffer: deque[bytes]) -> None:
        self.buffer = buffer
        self._current = b""

    def read(self) -> bytes:
        target = 3840  # 20ms at 48kHz stereo pcm16
        while len(self._current) < target and self.buffer:
            self._current += self.buffer.popleft()
        if not self._current:
            return b"\x00" * target
        chunk = self._current[:target]
        self._current = self._current[target:]
        if len(chunk) < target:
            chunk += b"\x00" * (target - len(chunk))
        return chunk
