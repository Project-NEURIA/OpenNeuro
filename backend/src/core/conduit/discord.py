from __future__ import annotations

import asyncio
import os
import threading
from collections import deque
from typing import NamedTuple, cast

import discord
from discord.sinks import PCMSink
from pydantic import BaseModel

from src.core.channel import Receiver, Sender
from src.core.component import PrimitiveComponent
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


class DiscordIO(PrimitiveComponent[DiscordInputs, DiscordOutputs]):
    """Discord audio conduit that handles both input and output."""

    def __init__(self, config: DiscordConfig) -> None:
        super().__init__()
        self.config: DiscordConfig = config

        token = os.getenv(self.config.token_env_var)
        if not token:
            raise ValueError(
                f"Environment variable {self.config.token_env_var} must be set"
            )
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

        async def cleanup_guild_voice(
            guild_id: int, voice_client: discord.VoiceClient | None = None
        ) -> None:
            task = _playback_tasks.pop(guild_id, None)
            if task:
                task.cancel()
            _rings.pop(guild_id, None)
            _buffer.pop(guild_id, None)

            vc = voice_client or _voice_clients.pop(guild_id, None)
            if vc:
                try:
                    if vc.recording:
                        vc.stop_recording()
                except Exception:
                    pass
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass

        async def purge_stale_voice(guild: discord.Guild) -> None:
            """Remove any stale voice client from py-cord's internal state."""
            existing = guild.voice_client
            if existing is None:
                return
            print(f"[DiscordIO] Purging stale voice client for guild {guild.id}")
            try:
                await existing.disconnect(force=True)
            except Exception:
                pass
            # If disconnect didn't clean up the state, remove it manually
            if guild.voice_client is not None:
                try:
                    existing.cleanup()
                except Exception:
                    pass
            await asyncio.sleep(0.5)

        async def wait_voice_connected(
            vc: discord.VoiceClient, timeout: float = 15.0
        ) -> bool:
            """Wait for the VoiceClient's internal _connected threading.Event."""
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, vc._connected.wait, timeout)

        async def finished_recording(_sink: PCMSink, *_args: object) -> None:
            return None

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
            ring: deque[bytes] = deque(maxlen=2000)
            if _active_discord_io:
                ring = deque(maxlen=_active_discord_io.max_frames)
            _rings[gid] = ring

            channel = member.voice.channel
            if channel is None:
                await ctx.respond("Join a voice channel first")
                return

            await ctx.defer()

            vc: discord.VoiceClient | None = None
            try:
                await purge_stale_voice(ctx.guild)

                vc = cast(
                    discord.VoiceClient,
                    await channel.connect(timeout=30.0, reconnect=False),
                )

                # Wait for the voice WS handshake to actually complete
                connected = await wait_voice_connected(vc, timeout=15.0)
                print(
                    f"[DiscordIO] Voice handshake for guild {gid}: "
                    f"connected={connected}, is_connected={vc.is_connected()}"
                )
                if not connected:
                    raise TimeoutError(
                        "Voice WebSocket handshake did not complete. "
                        "Ensure PyNaCl is installed (pip install PyNaCl) "
                        "and the bot can reach Discord voice servers."
                    )

                _voice_clients[gid] = vc
                sink = _DiscordSink()
                vc.start_recording(sink, finished_recording)

                _buffer[gid] = deque(maxlen=1000)
                if gid not in _playback_tasks or _playback_tasks[gid].done():
                    _playback_tasks[gid] = asyncio.create_task(self._playback_loop(gid))

                await ctx.followup.send("Connected")
            except Exception as e:
                await cleanup_guild_voice(gid, vc)
                err_str = str(e)
                if "4017" in err_str:
                    msg = (
                        "This voice channel requires end-to-end encryption "
                        "(DAVE protocol) which we currently cannot support. Kinda fucked."
                    )
                    print(f"[DiscordIO] Join failed for guild {gid}: {msg}")
                    await ctx.followup.send(msg)
                else:
                    print(f"[DiscordIO] Join failed for guild {gid}: {e}")
                    await ctx.followup.send(f"Failed to join voice channel: {e}")

        @bot.slash_command(name="leave", guild_ids=self.config.guild_ids or None)
        async def leave(ctx: discord.ApplicationContext) -> None:
            if ctx.guild is None:
                await ctx.respond("Must be used in a guild")
                return

            gid = ctx.guild.id
            await cleanup_guild_voice(gid)
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
            # Find the guild id for this sink if needed, but we can just use ring
            # Actually, _DiscordSink doesn't know its guild unless we tell it.
            # But the ring is global anyway.
            # Wait, the ring should be per-guild.
            # Let's simplify and just send to the active IO.
            _active_discord_io._output_audio.send(
                AudioFrame.new(
                    data=data,
                    sample_rate=48000,  # Discord is always 48k
                    channels=2,  # Discord is always stereo
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
