#!/usr/bin/env python3
"""Generate a deterministic RGB PPM image without external dependencies."""
import argparse
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--output", default="input.ppm")
    args = parser.parse_args()

    if args.width < 3 or args.height < 3:
        raise SystemExit("Width and height must be at least 3.")

    with open(args.output, "wb") as image:
        image.write(f"P6\n{args.width} {args.height}\n255\n".encode())
        for y in range(args.height):
            for x in range(args.width):
                checker = ((x // 48) + (y // 48)) % 2
                wave = int(127.5 + 127.5 * math.sin(x * 0.025) *
                           math.cos(y * 0.025))
                r = (x * 255 // args.width + checker * 50) % 256
                g = (y * 255 // args.height + wave // 3) % 256
                b = (wave + checker * 80) % 256
                image.write(bytes((r, g, b)))

    print(f"Generated {args.output}: {args.width}x{args.height}")

if __name__ == "__main__":
    main()
