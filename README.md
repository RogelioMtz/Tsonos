# Tsonos — Audio Device Tester

A small command-line utility to list and optionally test audio input/output devices using the Python sounddevice library. Designed for quick diagnostics: list device properties, play a test tone to outputs, record short samples from inputs and play them back.

## Features

- List system audio devices (name, host API, input/output channels, samplerate, default flags)
- Sort device list (by index, name, input or output channel count)
- Play a test tone to an output device (single or all outputs)
- Record a short sample from an input device (single or all inputs), compute RMS/peak and play it back
- Optional interactive test menu after listing devices

## Requirements

- Python 3.7+
- sounddevice
- numpy

Install dependencies:

```bash
py -3 -m pip install --upgrade pip
py -3 -m pip install sounddevice numpy
```
