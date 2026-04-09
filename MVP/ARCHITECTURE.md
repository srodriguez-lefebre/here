# MVP Architecture

This document explains the current architecture of the MVP inside `MVP/`.

The goal of the MVP is simple:

- record microphone audio
- record system audio
- optionally combine both sources
- transcribe the resulting session
- save one final text file

## Scope

This is not a full product architecture. It is the architecture of a narrow CLI MVP.

The current design prioritizes:

- short path from command to transcript
- practical support for Windows combined capture
- support for long recordings through chunking
- low operational complexity

## Entry Points

The CLI entrypoints are defined in [pyproject.toml](/C:/Users/savar/Documents/proyects/here/MVP/pyproject.toml):

- `record`
- `here`

The main CLI implementation lives in [cli.py](/C:/Users/savar/Documents/proyects/here/MVP/src/cli.py).

User-facing commands:

- `record`
- `record mic`
- `record os`

Behavior:

- `record` captures microphone + system audio together
- `record mic` captures microphone only
- `record os` captures system audio only

## High-Level Flow

The end-to-end flow is:

1. CLI command is invoked
2. recording service captures one or more audio sources into temporary WAV files
3. the recording session is normalized into a single working audio stream
4. the normalized stream is split into chunks when needed
5. each chunk is transcribed
6. chunk transcripts are merged
7. the final text is written to `transcriptions/YYYYMMDD_HHMMSS.txt`
8. temporary audio files are deleted

## Module Layout

The code is intentionally split into three main domains:

- `recording`
- `audio`
- `transcription`

Thin compatibility wrappers still exist at the top level:

- [recorder.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recorder.py)
- [chunking.py](/C:/Users/savar/Documents/proyects/here/MVP/src/chunking.py)
- [transcriber.py](/C:/Users/savar/Documents/proyects/here/MVP/src/transcriber.py)

Those files mainly re-export the newer internal modules.

## Recording Layer

Recording code lives under [recording/](/C:/Users/savar/Documents/proyects/here/MVP/src/recording).

### [models.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recording/models.py)

Defines the main recording data structures:

- `RecordedAudioSource`
- `RecordingSession`

`RecordingSession` is the main object passed from capture into transcription.

It contains one or more recorded source WAVs plus metadata such as:

- path
- sample rate
- channel count
- frame count
- source label

### [shared.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recording/shared.py)

Contains shared helpers for:

- temporary WAV creation
- temporary `SoundFile` opening
- building a single-source session
- safe writer cleanup

### [linux.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recording/linux.py)

Handles Linux/WSL capture.

Responsibilities:

- microphone capture through `sounddevice`
- system-audio capture through PulseAudio monitor source
- `pactl` integration
- temporary WAV writing

Linux system-audio capture depends on PulseAudio being available and correctly configured.

### [windows.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recording/windows.py)

Handles Windows capture.

Responsibilities:

- default microphone device discovery
- default WASAPI loopback device discovery
- Windows stream opening through `pyaudiowpatch`
- microphone-only capture
- system-only capture
- combined microphone + system capture
- pacing captured streams against wall clock time
- filling silence when a stream does not provide frames for a given interval

This is the most complex file in the codebase because it contains the platform-specific capture behavior.

### [service.py](/C:/Users/savar/Documents/proyects/here/MVP/src/recording/service.py)

This is the public orchestration layer for recording.

It decides:

- which backend to use
- which function to call for each command
- whether combined capture is allowed on the current platform

## Audio Layer

Audio processing code lives under [audio/](/C:/Users/savar/Documents/proyects/here/MVP/src/audio).

This layer is separate from capture and separate from the OpenAI client.

### [models.py](/C:/Users/savar/Documents/proyects/here/MVP/src/audio/models.py)

Defines:

- `ChunkingConfig`
- `ChunkWindow`

These are the internal configuration and planning objects for long-recording processing.

### [mix.py](/C:/Users/savar/Documents/proyects/here/MVP/src/audio/mix.py)

Responsible for audio normalization and source mixing.

Main responsibilities:

- downmix to mono
- resample to target sample rate
- align source lengths in target-frame space
- compute gain balancing per source
- create a normalized mixed working WAV from a `RecordingSession`

This is the stage that converts one-or-more captured source WAVs into one single transcription-ready session.

### [chunking.py](/C:/Users/savar/Documents/proyects/here/MVP/src/audio/chunking.py)

Responsible for chunk planning and transcript stitching helpers.

Main responsibilities:

- determine chunk windows from a byte budget
- apply overlap between chunks
- render chunk WAVs from a normalized session
- generate prompt tails when the model supports prompts
- merge transcript text across chunks
- deduplicate overlap using normalized word matching

The current stitching logic is text-based, not fully speaker-aware.

## Transcription Layer

Transcription code lives under [transcription/](/C:/Users/savar/Documents/proyects/here/MVP/src/transcription).

### [client.py](/C:/Users/savar/Documents/proyects/here/MVP/src/transcription/client.py)

Responsible for OpenAI integration.

Main responsibilities:

- build the API client
- resolve active transcription/cleanup models
- upload audio files for transcription
- select the correct response format
- support diarized transcription output
- extract transcript text from API responses
- run optional cleanup

Important detail:

The current default transcription model is `gpt-4o-transcribe-diarize`.

When the model is diarized:

- the request uses `diarized_json`
- `chunking_strategy="auto"` is sent
- prompt carryover is not used

### [service.py](/C:/Users/savar/Documents/proyects/here/MVP/src/transcription/service.py)

Responsible for end-to-end transcription orchestration.

Main responsibilities:

- transcribe a single audio file
- transcribe a multi-source recorded session
- normalize the session before chunking
- render and transcribe chunks in order
- merge all chunk transcripts
- apply optional cleanup at the end

This module is the bridge between:

- recording output
- audio normalization/chunking
- OpenAI transcription

## CLI Layer

[cli.py](/C:/Users/savar/Documents/proyects/here/MVP/src/cli.py) is intentionally small.

Its responsibilities are:

- expose commands with Typer
- resolve the output directory
- call the correct recording function
- call the transcription pipeline
- write the final transcript file
- translate runtime errors into CLI-friendly failures

This file does not contain platform logic, chunking logic, or OpenAI request construction.

## Configuration

Configuration lives in [settings.py](/C:/Users/savar/Documents/proyects/here/MVP/src/config/settings.py).

Important settings:

- `OPENAI_API_KEY`
- `TRANSCRIPTIONS_DIR`
- `PULSE_SERVER`
- `TRANSCRIPTION_MODEL`
- `CLEANUP_MODEL`
- `CLEANUP_ENABLED`

Settings are cached for the lifetime of the process to avoid repeated parsing of `.env`.

## Temporary Files

The MVP uses temporary WAV files heavily.

That is intentional.

Reasons:

- keeps recording memory usage bounded
- makes long recordings practical
- simplifies cross-stage processing
- allows chunk rendering without holding the entire session in RAM

Temporary audio is deleted after transcription completes.

The final user-visible output is only the transcript `.txt`.

## Chunking Strategy

The current chunking architecture is:

1. capture raw source WAVs
2. build a normalized mixed WAV
3. split the normalized WAV into local chunks
4. transcribe each chunk
5. merge chunk transcripts into one final text

The model is also allowed to do its own internal chunk handling via `chunking_strategy="auto"` for diarized transcription.

So the current design is a combination of:

- app-level chunking
- model-level chunk handling

This is acceptable for the MVP, but not the final form of a more advanced system.

## Current Limitations

The architecture is valid for the MVP, but there are still limitations:

- combined `record` is Windows-only
- transcript stitching is still text-based
- chunk merging is not fully speaker-aware
- cleanup is optional and off by default
- there is no search or retrieval layer
- there is no speaker identity management beyond what the transcription model provides
