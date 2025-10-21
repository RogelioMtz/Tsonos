import sys
import argparse
import json

try:
    import sounddevice as sd
except ImportError:
    print("Please install 'sounddevice' first: py -m pip install sounddevice")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Please install 'numpy' first: py -m pip install numpy")
    sys.exit(1)


def get_default_indices():
    d = sd.default.device
    if isinstance(d, tuple):
        return d
    if d is None:
        return (None, None)
    return (d, d)


def format_device(d, idx, default_in, default_out, show_sr):
    host = d.get("hostapi")
    host_name = hostapis_map.get(host, "Unknown API")
    in_ch = d.get("max_input_channels", 0)
    out_ch = d.get("max_output_channels", 0)
    sr = d.get("default_samplerate") if show_sr else None
    marks = []
    if idx == default_in:
        marks.append("default input")
    if idx == default_out:
        marks.append("default output")
    mark = f" ({', '.join(marks)})" if marks else ""
    sr_str = f" | sr: {int(sr) if sr else 'N/A'}" if show_sr else ""
    return f"  [{idx}] {d.get('name','<unknown>')} - {host_name} | in:{in_ch} out:{out_ch}{sr_str}{mark}"


def _get_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    global hostapis_map
    hostapis_map = {i: ha.get("name", f"API#{i}") for i, ha in enumerate(hostapis)}
    return devices


def test_output_device(index, duration=2.0, freq=1000.0, amp=0.2):
    try:
        dev = sd.query_devices(index)
    except Exception as e:
        print(f"[out {index}] cannot query device: {e}")
        return False
    out_ch = dev.get("max_output_channels", 0)
    if out_ch <= 0:
        print(f"[out {index}] no output channels, skipping")
        return False
    sr = int(dev.get("default_samplerate") or sd.default.samplerate or 44100)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Use up to 2 channels for the test tone (mono duplicated if needed)
    tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if out_ch >= 2:
        data = np.column_stack([tone, tone])
    else:
        data = tone
    try:
        print(f"[out {index}] playing {freq}Hz tone for {duration}s (sr={sr})")
        sd.play(data, samplerate=sr, device=index)
        sd.wait()
        print(f"[out {index}] finished")
        return True
    except Exception as e:
        print(f"[out {index}] playback failed: {e}")
        return False


def test_input_device(index, duration=2.0):
    try:
        dev = sd.query_devices(index)
    except Exception as e:
        print(f"[in  {index}] cannot query device: {e}")
        return False
    in_ch = dev.get("max_input_channels", 0)
    if in_ch <= 0:
        print(f"[in  {index}] no input channels, skipping")
        return False
    sr = int(dev.get("default_samplerate") or sd.default.samplerate or 44100)
    channels = min(in_ch, 2)  # record 1 or 2 channels for quick test
    frames = int(sr * duration)
    try:
        print(f"[in  {index}] recording {duration}s (sr={sr}, ch={channels}) ...")
        rec = sd.rec(frames, samplerate=sr, channels=channels, dtype="float32", device=index)
        sd.wait()
        # compute RMS per channel
        if rec.ndim == 1:
            rec = rec[:, None]
        rms = np.sqrt(np.mean(rec.astype(np.float64) ** 2, axis=0))
        rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
        for ch, (r, db) in enumerate(zip(rms, rms_db), start=1):
            print(f"  channel {ch}: RMS={r:.6f}, dBFS={db:.1f} dB")
        peak = np.max(np.abs(rec))
        print(f"  peak amplitude: {peak:.6f}")

        try:
            default_out = get_default_indices()[1]
            print(f"[in  {index}] playing back recording on output device {default_out} ...")
            # apply a modest gain to avoid very loud playback or feedback
            gain = 0.8
            play_data = (rec * gain).astype(np.float32)
            sd.play(play_data, samplerate=sr, device=default_out)
            sd.wait()
            print(f"[in  {index}] playback finished")
        except Exception as e:
            print(f"[in  {index}] playback failed: {e}")

        # delete recording buffer before returning to free memory / avoid reuse
        try:
            del rec
        except Exception:
            pass

        return True
    except Exception as e:
        print(f"[in  {index}] recording failed: {e}")
        return False


def list_devices(sort_by="index", json_out=False, show_sr=False):
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_in, default_out = get_default_indices()
    except Exception as e:
        print(f"Failed to query audio devices: {e}")
        sys.exit(1)

    global hostapis_map
    hostapis_map = {i: ha.get("name", f"API#{i}") for i, ha in enumerate(hostapis)}

    indexed = [(i, d) for i, d in enumerate(devices)]
    if sort_by == "name":
        indexed.sort(key=lambda t: (t[1].get("name") or "").lower())
    elif sort_by == "in":
        indexed.sort(key=lambda t: -t[1].get("max_input_channels", 0))
    elif sort_by == "out":
        indexed.sort(key=lambda t: -t[1].get("max_output_channels", 0))
    # default is index order

    if json_out:
        out = []
        for i, d in indexed:
            out.append(
                {
                    "index": i,
                    "name": d.get("name"),
                    "hostapi": hostapis_map.get(d.get("hostapi")),
                    "max_input_channels": d.get("max_input_channels", 0),
                    "max_output_channels": d.get("max_output_channels", 0),
                    "default_samplerate": d.get("default_samplerate"),
                    "is_default_input": i == default_in,
                    "is_default_output": i == default_out,
                }
            )
        print(json.dumps(out, indent=2))
        return

    print("Audio input devices:")
    for i, d in indexed:
        if d.get("max_input_channels", 0) > 0:
            print(format_device(d, i, default_in, default_out, show_sr))

    print("\nAudio output devices:")
    for i, d in indexed:
        if d.get("max_output_channels", 0) > 0:
            print(format_device(d, i, default_in, default_out, show_sr))


def interactive_test(devices, args):
    """Simple interactive menu to run tests after listing devices."""
    print("\nInteractive test mode. Press Enter to accept defaults or 'q' to quit.")
    while True:
        print("\nOptions:")
        print("  1) Test single output by index")
        print("  2) Test single input by index")
        print("  3) Test all outputs")
        print("  4) Test all inputs")
        print("  q) Quit")
        choice = input("Select option [q]: ").strip().lower()
        if choice in ("q", ""):
            break
        if choice == "1":
            s = input("Output device index: ").strip()
            try:
                idx = int(s)
            except Exception:
                print("Invalid index")
                continue
            dur = input(f"Duration seconds [{args.duration}]: ").strip() or str(args.duration)
            freq = input(f"Tone freq Hz [{args.freq}]: ").strip() or str(args.freq)
            amp = input(f"Amp 0..1 [{args.amp}]: ").strip() or str(args.amp)
            try:
                test_output_device(idx, duration=float(dur), freq=float(freq), amp=float(amp))
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "2":
            s = input("Input device index: ").strip()
            try:
                idx = int(s)
            except Exception:
                print("Invalid index")
                continue
            dur = input(f"Duration seconds [{args.duration}]: ").strip() or str(args.duration)
            try:
                test_input_device(idx, duration=float(dur))
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "3":
            dur = input(f"Duration seconds [{args.duration}]: ").strip() or str(args.duration)
            freq = input(f"Tone freq Hz [{args.freq}]: ").strip() or str(args.freq)
            amp = input(f"Amp 0..1 [{args.amp}]: ").strip() or str(args.amp)
            try:
                for i, d in enumerate(devices):
                    if d.get("max_output_channels", 0) > 0:
                        test_output_device(i, duration=float(dur), freq=float(freq), amp=float(amp))
            except Exception as e:
                print(f"Error: {e}")
        elif choice == "4":
            dur = input(f"Duration seconds [{args.duration}]: ").strip() or str(args.duration)
            try:
                for i, d in enumerate(devices):
                    if d.get("max_input_channels", 0) > 0:
                        test_input_device(i, duration=float(dur))
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Unknown option")
    print("Exiting interactive test mode.")


def main(argv=None):
    p = argparse.ArgumentParser(description="List and test audio devices (sounddevice)")
    p.add_argument("--json", action="store_true", help="Output device list as JSON")
    p.add_argument(
        "--sort",
        choices=["index", "name", "in", "out"],
        default="index",
        help="Sort devices",
    )
    p.add_argument("--show-sr", action="store_true", help="Show default samplerate")

    # testing options
    p.add_argument("--test-output-index", type=int, help="Play test tone to output device index")
    p.add_argument("--test-input-index", type=int, help="Record short sample from input device index")
    p.add_argument("--test-all-outputs", action="store_true", help="Play test tone to all output devices")
    p.add_argument("--test-all-inputs", action="store_true", help="Record short sample from all input devices")
    p.add_argument("--duration", type=float, default=2.0, help="Duration in seconds for tests (default: 2.0)")
    p.add_argument("--freq", type=float, default=1000.0, help="Frequency for output test tone (Hz)")
    p.add_argument("--amp", type=float, default=0.2, help="Amplitude for output test tone (0.0-1.0)")
    args = p.parse_args(argv)

    # Always list devices first
    list_devices(sort_by=args.sort, json_out=args.json, show_sr=args.show_sr)

    # If explicit test flags were provided, run them non-interactively
    ran_tests = False
    if args.test_output_index is not None:
        ran_tests = True
        test_output_device(args.test_output_index, duration=args.duration, freq=args.freq, amp=args.amp)

    if args.test_input_index is not None:
        ran_tests = True
        test_input_device(args.test_input_index, duration=args.duration)

    if args.test_all_outputs:
        ran_tests = True
        devices = _get_devices()
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) > 0:
                test_output_device(i, duration=args.duration, freq=args.freq, amp=args.amp)

    if args.test_all_inputs:
        ran_tests = True
        devices = _get_devices()
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                test_input_device(i, duration=args.duration)

    # If no test flags given and running interactively, offer optional interactive testing
    if not ran_tests and sys.stdin.isatty():
        resp = input("\nRun tests now? [Y/N]: ").strip().lower()
        if resp == "y":
            devices = _get_devices()
            interactive_test(devices, args)


if __name__ == "__main__":
    main()