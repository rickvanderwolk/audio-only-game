#!/usr/bin/env python3

import math
import time
import threading
import numpy as np
import sounddevice as sd
import pygame

SAMPLE_RATE = 44100

_audio_lock = threading.Lock()
_audio_queue = []


def _audio_callback(outdata, frames, time_info, status):
    global _audio_queue

    with _audio_lock:
        outdata[:] = 0
        finished = []
        for i, (samples, pos) in enumerate(_audio_queue):
            remaining = len(samples) - pos
            to_copy = min(frames, remaining)
            if to_copy > 0:
                outdata[:to_copy] += samples[pos:pos + to_copy]
                _audio_queue[i] = (samples, pos + to_copy)
            if pos + to_copy >= len(samples):
                finished.append(i)
        for i in reversed(finished):
            _audio_queue.pop(i)

        np.clip(outdata, -0.9, 0.9, out=outdata)


_stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    channels=2,
    dtype=np.float32,
    callback=_audio_callback,
    blocksize=2048,
    latency='high'
)
_stream.start()


def queue(samples):
    with _audio_lock:
        _audio_queue.append((samples, 0))


def delay(duration):
    time.sleep(duration)


def synth(frequency, duration, volume=0.3, pan=0.0):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * math.pi * frequency * t)

    fade = int(SAMPLE_RATE * 0.02)
    wave[:fade] *= np.linspace(0, 1, fade)
    wave[-fade:] *= np.linspace(1, 0, fade)

    left = volume * (1 - max(0, pan))
    right = volume * (1 + min(0, pan))

    stereo = np.column_stack([wave * left, wave * right])
    return stereo.astype(np.float32)


def tone_a(x, y):
    freq = 200 + y * 600
    queue(synth(freq, 0.08, volume=0.4, pan=x))


def tone_b():
    t = np.linspace(0, 0.1, int(SAMPLE_RATE * 0.1), False)
    wave = np.sin(2 * math.pi * 600 * t) + 0.5 * np.sin(2 * math.pi * 900 * t)
    wave *= np.linspace(1, 0, len(wave))
    stereo = np.column_stack([wave * 0.3, wave * 0.3])
    queue(stereo.astype(np.float32))


def tone_c():
    t = np.linspace(0, 0.3, int(SAMPLE_RATE * 0.3), False)
    wave = np.sin(2 * math.pi * 150 * t)
    wave *= np.linspace(1, 0, len(wave))
    stereo = np.column_stack([wave * 0.4, wave * 0.4])
    queue(stereo.astype(np.float32))


def tone_d():
    t = np.linspace(0, 0.15, int(SAMPLE_RATE * 0.15), False)
    wave = np.sin(2 * math.pi * 200 * t) + 0.3 * np.sin(2 * math.pi * 250 * t)
    wave *= np.linspace(1, 0, len(wave))
    stereo = np.column_stack([wave * 0.3, wave * 0.3])
    queue(stereo.astype(np.float32))


def tone_e():
    duration = 0.25
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    freq = np.linspace(300, 100, len(t))
    wave = np.sin(2 * math.pi * freq * t / SAMPLE_RATE * np.arange(len(t)))
    wave *= np.linspace(1, 0, len(wave))
    stereo = np.column_stack([wave * 0.35, wave * 0.35])
    queue(stereo.astype(np.float32))


def melody_a():
    notes = [440, 554, 659]
    data = []
    for freq in notes:
        t = np.linspace(0, 0.1, int(SAMPLE_RATE * 0.1), False)
        wave = np.sin(2 * math.pi * freq * t)
        wave *= np.linspace(1, 0, len(wave))
        data.extend(wave * 0.3)
    stereo = np.column_stack([data, data])
    queue(stereo.astype(np.float32))
    delay(0.35)


def melody_b():
    notes = [523, 659, 784, 1047]
    data = []
    for freq in notes:
        t = np.linspace(0, 0.15, int(SAMPLE_RATE * 0.15), False)
        wave = np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(2 * math.pi * freq * 2 * t)
        wave *= np.linspace(1, 0.3, len(wave))
        data.extend(wave * 0.35)
    stereo = np.column_stack([data, data])
    queue(stereo.astype(np.float32))
    delay(0.65)


def melody_c():
    notes = [392, 349, 294, 220]
    data = []
    for freq in notes:
        t = np.linspace(0, 0.2, int(SAMPLE_RATE * 0.2), False)
        wave = np.sin(2 * math.pi * freq * t)
        wave *= np.linspace(1, 0, len(wave))
        data.extend(wave * 0.35)
    stereo = np.column_stack([data, data])
    queue(stereo.astype(np.float32))
    delay(0.85)


def main():
    pygame.init()
    pygame.joystick.init()

    print("Please connect a gamepad...")

    while pygame.joystick.get_count() == 0:
        pygame.time.wait(500)
        pygame.joystick.quit()
        pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Gamepad connected. Game starting shortly...")
    time.sleep(2)

    melody_a()
    time.sleep(1.0)

    px = 0.0
    py = 0.5
    vx = -0.012
    vy = 0.008

    s1 = 0
    s2 = 0

    last_t = time.time()
    ready = False
    clock = pygame.time.Clock()

    running = True
    while running:
        try:
            now = time.time()

            px += vx
            py += vy

            if py <= 0 or py >= 1:
                vy = -vy
                py = max(0, min(1, py))

            if px >= 0.85:
                ready = True

            if px >= 1.0:
                if ready:
                    tone_c()
                    delay(0.35)
                    tone_e()
                    s2 += 1
                px = 0.0
                vx = abs(vx) * -1
                ready = False
                time.sleep(0.5)

            if px <= -1.0:
                vx = abs(vx)
                vy = (np.random.random() - 0.5) * 0.03

            d = 1 - px
            interval = 0.1 + d * 0.3

            if now - last_t > interval:
                tone_a(px, py)
                last_t = now

            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    if ready:
                        tone_b()
                        s1 += 1
                        vx = -abs(vx)
                        ready = False
                        vx *= 1.05
                    else:
                        tone_d()

                elif event.type == pygame.QUIT:
                    running = False

            if s1 >= 5 or s2 >= 5:
                if s1 >= 5:
                    melody_b()
                else:
                    melody_c()
                time.sleep(2)
                s1 = 0
                s2 = 0
                px = 0.0
                py = 0.5
                vx = -0.012
                vy = 0.01
                ready = False
                melody_a()
                time.sleep(1.0)

            clock.tick(60)

        except KeyboardInterrupt:
            running = False

    _stream.stop()
    _stream.close()
    pygame.quit()


if __name__ == "__main__":
    main()
