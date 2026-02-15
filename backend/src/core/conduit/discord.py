from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from typing import TypedDict, cast

import discord
from discord.sinks import Sink

from src.core.component import Component
from src.core.channel import Channel
from src.core.frames import AudioFrame, InterruptFrame, AudioDataFormat
from src.core.config import BaseConfig

# Global Discord bot instance and event loop
_discord_bot: discord.Bot | None = None
_discord_loop: asyncio.AbstractEventLoop | None = None
_discord_thread: threading.Thread | None = None
_discord_running = False
_discord_lock = threading.Lock()


class DiscordConfig(BaseConfig):
    token: str | None = None
    sample_rate: int = 48000
    channels: int = 2
    audio_buffer_seconds: int = 64
    guild_ids: list[int] = []


class DiscordOutputs(TypedDict):
    audio: Channel[AudioFrame]


class DiscordIO(
    Component[[Channel[AudioFrame], Channel[InterruptFrame]], DiscordOutputs]
):
    """Discord audio conduit that handles both input and output."""

    def __init__(self, config: DiscordConfig | None = None) -> None:
        super().__init__(config or DiscordConfig())
        self.config: DiscordConfig

        self.token = self.config.token or os.getenv("DISCORD_TOKEN")
        if not self.token:
            raise ValueError(
                "Discord token must be provided in config or DISCORD_TOKEN env var"
            )

        self.max_frames = self.config.audio_buffer_seconds * 50  # 20ms frames
        self._output_audio = Channel[AudioFrame](name="audio")

        self._rings: dict[int, deque[bytes]] = {}
        self._voice_clients: dict[int, discord.VoiceClient] = {}
        self._buffer: dict[int, deque[bytes]] = {}
        self._playback_tasks: dict[int, asyncio.Task] = {}

        print(f"[DiscordIO] DiscordIO initialized, guild_ids: {self.config.guild_ids}")

        self._ensure_discord_running()

    def get_output_channels(self) -> DiscordOutputs:
        return {"audio": self._output_audio}

    def _ensure_discord_running(self) -> None:
        global _discord_bot, _discord_loop, _discord_thread, _discord_running
        with _discord_lock:
            if _discord_running:
                return

            def run_discord():
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
        async def on_ready():
            print(f"[DiscordIO] Bot ready: {bot.user}")

        @bot.slash_command(name="join", guild_ids=self.config.guild_ids or None)
        async def join(ctx: discord.ApplicationContext):
            member = ctx.author
            if not isinstance(member, discord.Member) or member.voice is None:
                await ctx.respond("Join a voice channel first")
                return

            if ctx.guild is None:
                await ctx.respond("Must be used in a guild")
                return

            gid = ctx.guild.id
            ring: deque[bytes] = deque(maxlen=self.max_frames)
            self._rings[gid] = ring

            channel = member.voice.channel
            if channel is None:
                await ctx.respond("Join a voice channel first")
                return

            vc = cast(discord.VoiceClient, await channel.connect())
            sink = _DiscordSink(
                self._output_audio, ring, self.config.sample_rate, self.config.channels
            )

            vc.start_recording(sink, lambda *_args: None)

            self._voice_clients[gid] = vc
            self._buffer[gid] = deque(maxlen=1000)
            if _discord_loop:
                task = asyncio.run_coroutine_threadsafe(
                    self._playback_loop(gid), _discord_loop
                )
                self._playback_tasks[gid] = cast(asyncio.Task[None], task)
            await ctx.respond("Connected")

        @bot.slash_command(name="leave", guild_ids=self.config.guild_ids or None)
        async def leave(ctx: discord.ApplicationContext):
            if ctx.guild is None:
                await ctx.respond("Must be used in a guild")
                return

            gid = ctx.guild.id
            vc = ctx.guild.voice_client
            if vc:
                await vc.disconnect()
                self._rings.pop(gid, None)
                self._voice_clients.pop(gid, None)
                self._buffer.pop(gid, None)
                task = self._playback_tasks.pop(gid, None)
                if task:
                    task.cancel()
            await ctx.respond("Disconnected")

    async def _playback_loop(self, guild_id: int):
        vc = self._voice_clients[guild_id]
        buffer = self._buffer[guild_id]
        source = _DiscordAudioSource(buffer)
        vc.play(source)
        while not self.stop_event.is_set() and vc.is_connected():
            await asyncio.sleep(1.0)

    def run(
        self,
        audio: Channel[AudioFrame] | None = None,
        interrupt: Channel[InterruptFrame] | None = None,
    ) -> None:
        print("[DiscordIO] Starting Discord processing")

        if interrupt:

            def handle_interrupts():
                for frame in interrupt.stream(self):
                    if frame is None:
                        break
                    for buffer in self._buffer.values():
                        buffer.clear()

            threading.Thread(target=handle_interrupts, daemon=True).start()

        if audio:
            for frame in audio.stream(self):
                if frame is None:
                    break

                # Use AudioFrame.get for resampling/reformatting
                pcm_data = frame.get(
                    sample_rate=self.config.sample_rate,
                    num_channels=self.config.channels,
                    data_format=AudioDataFormat.PCM16,
                )

                for buffer in self._buffer.values():
                    buffer.append(pcm_data)

        print("[DiscordIO] Discord processing stopped")


class _DiscordSink(Sink):
    def __init__(
        self, output: Channel[AudioFrame], ring: deque[bytes], sr: int, ch: int
    ):
        super().__init__()
        self.output = output
        self.ring = ring
        self.sr = sr
        self.ch = ch

    def write(self, data: bytes, user):
        self.ring.append(data)
        self.output.send(
            AudioFrame(
                display_name="discord_audio",
                data=data,
                sample_rate=self.sr,
                channels=self.ch,
            )
        )


class _DiscordAudioSource(discord.AudioSource):
    def __init__(self, buffer: deque[bytes]):
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
