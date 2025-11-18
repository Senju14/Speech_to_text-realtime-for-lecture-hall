import sounddevice as sd
import numpy as np
import queue
import struct
import json
import time
import sys
import asyncio
import websockets


SERVER_WS = "ws://127.0.0.1:8012/ws" # change to your server address
SR = 16000
CHUNK_SAMPLES = 512 # 512 samples ≈ 32 ms
BUFFER = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️", status, file=sys.stderr)
    mono = np.squeeze(indata)
    BUFFER.put(mono.copy())




async def send_loop(uri):
    async with websockets.connect(uri, max_size=None) as ws:
        print("Connected to server", uri)


        # start audio stream
        stream = sd.InputStream(samplerate=SR, channels=1, dtype='int16', callback=audio_callback, blocksize=CHUNK_SAMPLES)
        stream.start()

        async def recv_loop():
            try:
                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    ttype = data.get("type")
                    if ttype == "realtime":
                        print(f"[partial] {data.get('text')}")
                    elif ttype == "fullSentence":
                        print(f"[final] {data.get('text')}")
                    elif ttype == "vad_start":
                        print("[VAD] start")
                    elif ttype == "vad_stop":
                        print("[VAD] stop")
                    else:
                        print("[srv]", data)
            except websockets.ConnectionClosed:
                print("Connection closed by server")


        recv_task = asyncio.create_task(recv_loop())
        try:
            while True:
                try:
                    chunk = BUFFER.get(timeout=1.0)
                except Exception:
                    await asyncio.sleep(0.01)
                    continue
                pcm_bytes = chunk.astype(np.int16).tobytes()
                metadata = {"sampleRate": SR}
                meta_json = json.dumps(metadata)
                meta_len = len(meta_json)
                message = struct.pack("<I", meta_len) + meta_json.encode("utf-8") + pcm_bytes
                await ws.send(message)
        except KeyboardInterrupt:
            print("Stopping client")
        finally:
            stream.stop()
            recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(send_loop(SERVER_WS))
