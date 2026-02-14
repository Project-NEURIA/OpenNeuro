from __future__ import annotations

import discord
import asyncio
import numpy as np
from discord.sinks import Sink
from collections import deque
import threading

from openneuro.core.component import Component
from openneuro.core.channel import Channel
from openneuro.core.frames import AudioFrame, InterruptFrame
from openneuro.config import get_config_value
import os

# Global Discord bot instance and event loop
_discord_bot: discord.Bot | None = None
_discord_loop: asyncio.AbstractEventLoop | None = None
_discord_thread: threading.Thread | None = None
_discord_running = False
_discord_lock = threading.Lock()


class DiscordIO(Component[Channel[AudioFrame]]):
    """Discord audio conduit that handles both input and output through voice channels."""
    
    def __init__(
        self,
        *,
        token: str | None = None,
        sample_rate: int = 48000,
        channels: int = 2,
        audio_buffer_seconds: int = 64,
    ) -> None:
        super().__init__()
        
        if token is None:
            token = os.environ['DISCORD_TOKEN']
            print("[DiscordIO] Using DISCORD_TOKEN {}".format(token))
        
        self.token = token
        self.max_frames = audio_buffer_seconds * 50  # 20ms frames
        self._sample_rate = sample_rate
        self._channels = channels
        
        # Get guild IDs from config
        guild_ids = get_config_value("discord.guild_ids", [])
        self.guild_ids = guild_ids if guild_ids else None
        
        # Discord bot will be created in run() method to avoid event loop issues
        self.bot: discord.Bot | None = None
        
        self._input_channel: Channel[AudioFrame] | None = None
        self._output_channel = Channel[AudioFrame]()
        
        self._rings: dict[int, deque[bytes]] = {}
        self._voice_clients: dict[int, discord.VoiceClient] = {}
        self._buffer: dict[int, deque[bytes]] = {}
        self._playback_tasks: dict[int, asyncio.Task] = {}
        
        # Ensure Discord bot is running
        self._ensure_discord_running()
    
    def _ensure_discord_running(self) -> None:
        """Ensure Discord bot is running (global singleton)."""
        global _discord_bot, _discord_loop, _discord_thread, _discord_running
        
        with _discord_lock:
            if _discord_running:
                # Discord is already running, just reference it
                self.bot = _discord_bot
                return
            
            # Start Discord bot in dedicated thread
            def run_discord():
                global _discord_bot, _discord_loop, _discord_running
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                _discord_loop = loop
                
                intents = discord.Intents.default()
                bot = discord.Bot(intents=intents)
                _discord_bot = bot
                
                # Register handlers for this instance
                self._register_handlers_for_bot(bot)
                
                _discord_running = True
                print("[DiscordIO] Discord bot started in dedicated thread")
                
                try:
                    loop.run_until_complete(bot.start(self.token))
                except Exception as e:
                    print(f"Discord bot error: {e}")
                finally:
                    _discord_running = False
                    _discord_bot = None
                    _discord_loop = None
                    loop.close()
            
            _discord_thread = threading.Thread(target=run_discord, daemon=True)
            _discord_thread.start()
            
            # Wait for bot to be ready
            while _discord_bot is None:
                threading.Event().wait(0.1)
            
            self.bot = _discord_bot
    
    def get_output_channels(self) -> tuple[Channel[AudioFrame]]:
        return (self._output_channel,)
    
    def set_input_channels(self, input_channel: Channel[AudioFrame]) -> None:
        self._input_channel = input_channel
    
    @classmethod
    def get_input_names(cls) -> list[str]:
        """Returns descriptive name for the input slot."""
        return ["audio_input"]
    
    def _register_handlers_for_bot(self, bot: discord.Bot) -> None:
        """Register Discord handlers for a specific bot instance."""
        @bot.event
        async def on_ready() -> None:
            print(f"Discord IO ready: {bot.user}")
        
        @bot.slash_command(
            name="join",
            description="Join voice and start capturing everyone's audio",
            guild_ids=self.guild_ids,
        )
        async def join(ctx: discord.ApplicationContext) -> None:
            if ctx.author.voice is None:
                await ctx.respond("Join a voice channel first")
                return
            
            gid = ctx.guild.id
            ring = deque(maxlen=self.max_frames)
            self._rings[gid] = ring
            
            vc = await ctx.author.voice.channel.connect()
            sink = _DiscordSink(self._output_channel, ring, self._sample_rate, self._channels)
            
            async def finished(sink: Sink, channel, *args) -> None:
                return
            
            vc.start_recording(sink, finished, ctx.channel)
            
            # Start playback for this guild if we have input
            if self._input_channel and gid not in self._voice_clients:
                self._voice_clients[gid] = vc
                self._buffer[gid] = deque(maxlen=1000)
                if _discord_loop:
                    self._playback_tasks[gid] = asyncio.run_coroutine_threadsafe(
                        self._playback_loop(gid), _discord_loop
                    )
            
            await ctx.respond("Connected and capturing audio")
        
        @bot.slash_command(
            name="leave",
            description="Stop capturing and leave voice",
            guild_ids=self.guild_ids,
        )
        async def leave(ctx: discord.ApplicationContext) -> None:
            gid = ctx.guild.id
            vc = ctx.guild.voice_client
            if vc and vc.recording:
                vc.stop_recording()
            if vc:
                await vc.disconnect()
                
                # Clean up guild-specific data
                self._rings.pop(gid, None)
                self._voice_clients.pop(gid, None)
                self._buffer.pop(gid, None)
                
                # Cancel playback task
                task = self._playback_tasks.pop(gid, None)
                if task and _discord_loop:
                    asyncio.run_coroutine_threadsafe(task.cancel(), _discord_loop)
                    
            await ctx.respond("Disconnected")
    
    def _register_handlers(self) -> None:
        """Register handlers (legacy method for compatibility)."""
        if self.bot:
            self._register_handlers_for_bot(self.bot)
    
    async def _playback_loop(self, guild_id: int) -> None:
        """Audio playback loop for a specific guild."""
        vc = self._voice_clients[guild_id]
        buffer = self._buffer[guild_id]
        
        # Create a streaming audio source that reads from the buffer
        source = _DiscordAudioSource(buffer)
        
        # Start playing the source - it will continuously read from the buffer
        vc.play(source)
        
        # Keep the source alive while the component is running
        while not self.stop_event.is_set() and vc.is_connected():
            await asyncio.sleep(1.0)  # Check every second
    
    def run(self) -> None:
        """Process incoming audio frames. Discord bot is already running."""
        print("[DiscordIO] Starting audio processing...")
        
        # Process incoming audio frames
        if self._input_channel:
            print("Stuff")
            for frame in self._input_channel.stream(self.stop_event):
                if frame is None:
                    break
                
                if isinstance(frame, InterruptFrame):
                    # Clear buffers on interrupt
                    for buffer in self._buffer.values():
                        buffer.clear()
                    continue
                
                if not isinstance(frame, AudioFrame):
                    continue
                
                # Resample if needed
                if frame.sample_rate != self._sample_rate:
                    frame = frame.resample(self._sample_rate)
                
                # Convert channels if needed
                data = self._convert_channels(frame)
                
                # Add to all connected voice client buffers
                for buffer in self._buffer.values():
                    buffer.append(data)
        
        print("[DiscordIO] Audio processing stopped")
    
    async def _run_bot(self) -> None:
        """Run the Discord bot."""
        if self.bot is None:
            return
        try:
            await self.bot.start(self.token)
        except Exception as e:
            print(f"Discord bot error: {e}")
    
    def _convert_channels(self, frame: AudioFrame) -> bytes:
        """Convert audio to the expected channel configuration."""
        if frame.channels == self._channels:
            return frame.pcm16_data
        
        arr = np.frombuffer(frame.pcm16_data, dtype=np.int16)
        
        if frame.channels == 1 and self._channels == 2:
            # Mono to stereo
            stereo = np.repeat(arr, 2)
            return stereo.tobytes()
        elif frame.channels == 2 and self._channels == 1:
            # Stereo to mono
            mono = arr.reshape(-1, 2).mean(axis=1, dtype=np.int16)
            return mono.tobytes()
        
        return frame.pcm16_data


class _DiscordSink(Sink):
    """Internal Discord sink that converts voice data to AudioFrames."""
    
    def __init__(
        self, 
        output_channel: Channel[AudioFrame],
        ring: deque[bytes],
        sample_rate: int,
        channels: int
    ) -> None:
        super().__init__()
        self.output_channel = output_channel
        self.ring = ring
        self.sample_rate = sample_rate
        self.channels = channels
    
    def write(self, data: bytes, user) -> None:
        uid = user if isinstance(user, int) else user.id
        
        # Store in ring buffer for potential use
        self.ring.append(data)
        
        # Discord provides 48kHz stereo 16-bit PCM
        audio_frame = AudioFrame(
            frame_type_string="discord_audio",
            pcm16_data=data,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        
        self.output_channel.send(audio_frame)


class _DiscordAudioSource(discord.AudioSource):
    """Custom Discord audio source for streaming PCM data."""
    
    def __init__(self, buffer: deque[bytes]):
        self.buffer = buffer
        self._current_data = b""
        self.position = 0
    
    def read(self) -> bytes:
        # Discord expects 3840 bytes for 20ms at 48kHz stereo
        target = 3840
        
        # If we need more data, get it from buffer
        while len(self._current_data) < target and self.buffer:
            chunk = self.buffer.popleft()
            self._current_data += chunk
        
        if not self._current_data:
            # No data available, return silence
            return b"\x00" * target
        
        # Extract the chunk we need
        chunk = self._current_data[:target]
        self._current_data = self._current_data[target:]
        
        # Pad with silence if needed
        if len(chunk) < target:
            chunk += b"\x00" * (target - len(chunk))
        
        return chunk
