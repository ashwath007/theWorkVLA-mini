# India Egocentric VLA Data Engine

A complete open-source pipeline for capturing, processing, and fine-tuning Vision-Language-Action (VLA) models on egocentric headset recordings of workers performing real-world tasks in Indian industrial and domestic environments.

---

## Why This Project

Robotic systems that work in the physical world need vast amounts of diverse, high-quality training data. Simulating Indian workplace environments — with their language, tools, textures, lighting conditions, and human motion patterns — is prohibitively expensive. Instead, this project takes a data-centric approach:

1. Attach a lightweight headset to real workers
2. Record video, audio (Hindi/English speech), and IMU (head motion) simultaneously
3. Process the egocentric stream into LeRobot-compatible training episodes
4. Fine-tune VLA models on task-specific data

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Headset (Raspberry Pi / PC)                │
│  Camera (30fps) ──┐                                             │
│  Microphone ──────┼──▶  HeadsetRecorder  ──▶  /data/sessions/  │
│  MPU-9250 IMU ────┘         (src/capture)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Preprocessing Pipeline                      │
│                                                                 │
│  StreamSynchronizer ──▶ VideoPreprocessor (face blur + resize)  │
│                     ──▶ AudioPreprocessor (resample to 16kHz)   │
│                     ──▶ IMUPreprocessor   (gravity removal)     │
│                     ──▶ HDF5Writer        (LeRobot schema)      │
│                              (src/preprocess)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Segmentation & Labelling                    │
│                                                                 │
│  ActionSegmenter (optical flow / uniform windows)               │
│  HindiTranscriber (Whisper — supports Hindi + code-mixed)       │
│  LeRobotChunker   (produces per-episode HDF5)                   │
│                              (src/segmentation)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage & Distribution                      │
│                                                                 │
│  HDF5Store    (local episode management)                        │
│  HFUploader   (push to Hugging Face Hub)                        │
│                              (src/storage)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Model Training                              │
│                                                                 │
│  IndiaVLADataset  (PyTorch Dataset with augmentation)           │
│  IndiaVLAModel    (MobileNetV3 + TinyLlama + IMU-MLP + Fusion)  │
│  Trainer          (BC loss, cosine LR, TensorBoard)             │
│                              (src/training)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     REST API (FastAPI)                          │
│                                                                 │
│  /sessions    CRUD for recorded sessions                        │
│  /pipeline    trigger preprocess / segment / chunk              │
│  /training    start / monitor training jobs                     │
│                              (src/api)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Data Format — LeRobot v0.5 HDF5 Schema

Each episode HDF5 file contains:

```
/observation/images/front_camera   (T, 224, 224, 3) uint8   — BGR frames, faces blurred
/observation/audio                 (T, samples_per_frame) float32 — 16kHz mono
/observation/imu                   (T, 10) float32  — [ax ay az gx gy gz qx qy qz qw]
/action                            (T, 7) float32   — head pose delta [dx dy dz dqx dqy dqz dqw]
/language_instruction              string (Hindi/English/code-mixed)
/episode_index                     (T,) int64
/frame_index                       (T,) int64
/timestamp                         (T,) float64 — Unix epoch seconds
/metadata/                         attrs: session_id, fps, language, lerobot_version
```

### Model Architecture

```
images (B,3,H,W) ──▶ VisionEncoder (MobileNetV3-small) ─────────────────┐
                                                                          ▼
tokens (B,L)     ──▶ LanguageEncoder (TinyLlama embed) ──▶ CrossAttentionFusion ──▶ ActionHead ──▶ (B,7)
                                                                          ▲
imu    (B,10)    ──▶ IMUEncoder (3-layer MLP) ──────────────────────────┘
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone and configure
git clone https://github.com/your-org/india-vla-engine.git
cd india-vla-engine
cp .env.example .env
# Edit .env with your HF_TOKEN and paths

# 2. Start services
docker-compose up -d

# 3. Open API docs
open http://localhost:8000/docs
# Open Label Studio
open http://localhost:8080
```

### Option B — Local development

```bash
# 1. Create environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Set env vars
export DATA_DIR=/tmp/vla_data
export MODEL_OUTPUT_DIR=/tmp/vla_models

# 3. Generate synthetic test data
python scripts/create_synthetic_data.py \
    --num-sessions 20 \
    --frames-per-session 90 \
    --output-dir $DATA_DIR/synthetic

# 4. Run training on synthetic data
python -m src.training.train \
    --dataset-dir $DATA_DIR/synthetic \
    --output-dir $MODEL_OUTPUT_DIR \
    --epochs 5

# 5. Start the API
uvicorn src.api.main:app --reload
```

### Option C — Raspberry Pi (real headset)

```bash
sudo bash scripts/setup_rpi.sh

# Start recording (auto-starts on boot via systemd)
sudo systemctl start vla-recorder

# Or record manually
source .venv/bin/activate
vla-record record --duration 600 --session-name "factory-assembly-01"

# Process the session
export SESSION_DIR=$(ls -d /data/sessions/*/* | tail -1)
bash scripts/run_pipeline.sh "$SESSION_DIR" --upload
```

---

## Recording

```bash
# Basic recording (Ctrl+C to stop)
vla-record record

# Named session with 10-minute duration
vla-record record --session-name "cutting-task" --duration 600

# Simulated IMU for development
vla-record record --simulate-imu

# List all sessions
vla-record info
```

---

## Pipeline

### Programmatic

```python
from src.preprocess.sync   import StreamSynchronizer
from src.preprocess.video  import VideoPreprocessor
from src.preprocess.audio  import AudioPreprocessor
from src.preprocess.imu    import IMUPreprocessor
from src.preprocess.hdf5_writer import HDF5Writer

from src.segmentation.action_segmenter import ActionSegmenter
from src.segmentation.transcriber      import HindiTranscriber
from src.segmentation.lerobot_chunker  import LeRobotChunker

session_dir = "/data/sessions/2024-01-15-1030/abc123"

# 1. Synchronize and preprocess
syncer  = StreamSynchronizer()
aligned = syncer.align_streams(
    f"{session_dir}/video.mp4",
    f"{session_dir}/audio.wav",
    f"{session_dir}/imu.csv",
    f"{session_dir}/metadata.json",
)

# 2. Segment actions
segmenter = ActionSegmenter()
# Extract frames first (see VideoPreprocessor.extract_frames)
segments  = segmenter.segment_optical_flow(bgr_frames)

# 3. Transcribe speech
transcriber = HindiTranscriber(model_name="small")
transcripts = transcriber.transcribe(f"{session_dir}/audio.wav", language="hi")

# 4. Create LeRobot episodes
chunker  = LeRobotChunker()
episodes = list(chunker.chunk_session(hdf5_path, segments, transcripts))
chunker.save_episodes(episodes, f"{session_dir}/episodes")
```

### Via shell script

```bash
bash scripts/run_pipeline.sh /data/sessions/2024-01-15-1030/abc123
bash scripts/run_pipeline.sh /data/sessions/2024-01-15-1030/abc123 --upload
```

### Via REST API

```bash
# Trigger preprocessing
curl -X POST http://localhost:8000/pipeline/preprocess/<session_id>

# Check job status
curl http://localhost:8000/pipeline/status/<job_id>

# Trigger segmentation (after preprocess completes)
curl -X POST http://localhost:8000/pipeline/segment/<session_id>

# Trigger chunking
curl -X POST http://localhost:8000/pipeline/chunk/<session_id>
```

---

## Training

```bash
# Start training with config file
python -m src.training.train --config configs/training_config.yaml

# Or with CLI options
python -m src.training.train \
    --dataset-dir /data/episodes \
    --output-dir ./checkpoints \
    --epochs 50 \
    --batch-size 32

# Monitor with TensorBoard
tensorboard --logdir ./logs
```

### Upload to Hugging Face

```python
from src.storage.hf_uploader import HFUploader

uploader = HFUploader(
    repo_id="your-username/india-pov-vla-v1",
    token="hf_...",
)
uploader.upload_dataset("/data/episodes", split="train")
uploader.create_dataset_card()
```

---

## Project Structure

```
theWorkVLA-mini/
├── src/
│   ├── capture/
│   │   ├── recorder.py        # HeadsetRecorder (video + audio + IMU)
│   │   └── cli.py             # Typer CLI for recording
│   ├── preprocess/
│   │   ├── sync.py            # StreamSynchronizer
│   │   ├── video.py           # VideoPreprocessor (face blur, resize)
│   │   ├── audio.py           # AudioPreprocessor (resample, chunk)
│   │   ├── imu.py             # IMUPreprocessor (gravity removal, quaternion)
│   │   └── hdf5_writer.py     # HDF5Writer (LeRobot v0.5 schema)
│   ├── segmentation/
│   │   ├── action_segmenter.py # Optical flow / uniform segmentation
│   │   ├── transcriber.py      # HindiTranscriber (Whisper)
│   │   └── lerobot_chunker.py  # LeRobotChunker
│   ├── storage/
│   │   ├── hdf5_store.py       # HDF5Store (local episode management)
│   │   └── hf_uploader.py      # HFUploader (Hugging Face Hub)
│   ├── training/
│   │   ├── dataset.py          # IndiaVLADataset (PyTorch)
│   │   ├── model.py            # IndiaVLAModel (VLA architecture)
│   │   └── train.py            # Trainer + CLI
│   └── api/
│       ├── main.py             # FastAPI app
│       └── routes/
│           ├── sessions.py     # /sessions CRUD
│           ├── pipeline.py     # /pipeline triggers
│           └── training.py     # /training management
├── scripts/
│   ├── setup_rpi.sh            # Raspberry Pi setup
│   ├── run_pipeline.sh         # End-to-end shell pipeline
│   └── create_synthetic_data.py # Synthetic data generator
├── configs/
│   └── training_config.yaml
├── tests/
│   ├── test_preprocess.py
│   ├── test_segmentation.py
│   └── test_training.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── setup.py
```

---

## Running Tests

```bash
# All tests (offline-safe — no model downloads required)
pytest tests/ -v

# Specific module
pytest tests/test_preprocess.py -v
pytest tests/test_segmentation.py -v
pytest tests/test_training.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Privacy

All faces detected in recorded video are automatically blurred using OpenCV's
Haar cascade face detector **before** the data is stored or uploaded. This is
applied in `VideoPreprocessor.blur_faces()` and is enabled by default in the
preprocessing pipeline.

---

## Hardware Requirements

| Mode | Minimum | Recommended |
|------|---------|-------------|
| Recording (RPi) | Raspberry Pi 4 (4GB), USB webcam, USB microphone, MPU-9250 IMU | RPi 5 + CSI camera + I2S microphone |
| Preprocessing (PC) | 8GB RAM, 4-core CPU | 16GB RAM, NVIDIA GPU |
| Training | 16GB RAM, NVIDIA GPU (8GB VRAM) | A100 / H100 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face API token for dataset upload |
| `HF_DATASET_REPO` | — | HF dataset repository ID |
| `POSTGRES_URL` | — | PostgreSQL + pgvector connection string |
| `DATA_DIR` | `/data/sessions` | Root directory for session data |
| `MODEL_OUTPUT_DIR` | `/models` | Directory for trained model checkpoints |
| `WHISPER_MODEL` | `tiny` | Whisper model size for transcription |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO model for object detection (future) |

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository and create a feature branch: `git checkout -b feature/my-feature`
2. **Code style**: We use `black`, `isort`, and `ruff`. Run `make lint` before committing.
3. **Tests**: All new features must include pytest tests. Aim for > 80% coverage.
4. **Documentation**: Add docstrings to all classes and public methods.
5. **Offline-first**: Features should not require internet access to pass tests.
6. **Privacy**: Never commit real person face data. Use synthetic or blurred data.

### Development setup

```bash
pip install -e ".[dev]"
pre-commit install
```

### Commit conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat: add Hindi NLP tokenizer`
- `fix: correct audio chunk alignment`
- `docs: update architecture diagram`
- `test: add IMU preprocessing tests`

### Areas for contribution

- [ ] Real-time streaming mode (WebRTC or RTSP)
- [ ] Active learning loop (uncertainty sampling for labelling)
- [ ] Multi-camera support (stereo / depth)
- [ ] LoRA fine-tuning for larger LLM backbones
- [ ] SLAM integration for absolute head pose
- [ ] Hindi named-entity recognition for task labels
- [ ] Mobile app for remote monitoring of recordings

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{india_vla_engine_2024,
  title     = {India Egocentric VLA Data Engine},
  author    = {India VLA Contributors},
  year      = {2024},
  url       = {https://github.com/your-org/india-vla-engine},
  license   = {Apache-2.0}
}
```
