import asyncio
import websockets

async def handler(websocket, path):
    print("GOT CONNECTION:", path, websocket.request_headers)
    await websocket.send("HELLO")

start_server = websockets.serve(handler, "127.0.0.1", 8001)
asyncio.get_event_loop().run_until_complete(start_server)
print("Listening 8001...")
asyncio.get_event_loop().run_forever()
