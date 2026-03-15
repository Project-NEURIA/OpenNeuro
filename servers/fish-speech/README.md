### What is this?
A heavily modded fork of https://github.com/fishaudio/fish-speech

Mainly we do this to get a better local fishtts server:
- 4 bit quatization, VRAM usage from 21GB -> 10GB
- RTF 1.6 -> 0.3
- Latency from over 1000ms -> ~256ms via optimizations and chunk based streaming
- reference ids from local folder `ref_samples`
- stripped down to only server related code

It's not a perfect server but it gets the job done for now while we wait for official implementation to get better.

[WIP]
I would be great if we can provide the model with some context, from the last sentence for example. This will smooth out that that little chopply jump between a sentence and the next... hmmm... could put some thought on that, make it fully streaming yk?

### To run the server
```
pip install -e .  # or somegthing else idk, uv sync?

python run_server.py  # You should probably read this script, it's basically just running the actual command via subprocess.run in python.

# for an example client, try python demo_fish_streaming.py
```