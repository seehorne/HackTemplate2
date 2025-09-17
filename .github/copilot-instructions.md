# WhatsAI Web Client

**ALWAYS** reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

WhatsAI is an AI-powered web client designed for Blind and Low Vision users. It provides real-time screen processing using various AI processors (YOLO object detection, Florence-2 scene captioning, MediaPipe finger counting, etc.) through a WebSocket-based streaming interface.

## Working Effectively

### Bootstrap and Build
- **Install conda environments**:
  - Copy environment definitions: `cp -r resources/whatsai /tmp/ && cp -r resources/aws /tmp/`
  - Install whatsai environment: `bash resources/install_environments.sh whatsai`
  - **NEVER CANCEL**: Environment creation takes 4-6 minutes. Set timeout to 10+ minutes.
  - Install missing dependency: `conda run -n whatsai pip install google-genai`
  - Install aws environment: `bash resources/install_environments.sh aws` 
  - **NEVER CANCEL**: May fail in CI environments due to network timeouts. This is expected.

### Docker Build (Primary Method)
- **Prerequisites**: Docker Desktop with WSL2 integration, `.env` file with `GEMINI_API_KEY="your-api-key"`
- **Build command**: `docker compose build`
- **NEVER CANCEL**: Docker build takes 15-25 minutes due to conda environment setup and ML model downloads. Set timeout to 30+ minutes.
- **Start server**: `docker compose up`
- **Known issue**: May fail in CI environments due to SSL certificate verification issues with GitHub downloads.

### Local Development Setup (Alternative)
- **Prerequisites**: Conda installed, Python 3.10+
- **Environment setup**: Follow bootstrap steps above
- **Start main server**: `conda activate whatsai && uvicorn stream_wss:app --host 0.0.0.0 --port 8000`
- **Start alternative server**: `conda activate aws && uvicorn stream_sonic:app --host 0.0.0.0 --port 8000`

### Test the System
- **Client access**: Open `client/screen_wss.html` in browser
- **Server URL**: `ws://localhost:8000/ws` for local development
- **Test processors**: Use processor IDs 0 (basic), 2 (scene captioning), 4 (object detection)
- **Verify connectivity**: `curl http://localhost:8000/processors`

## Validation

### Manual Testing Requirements
- **ALWAYS** test end-to-end functionality after making changes
- **Required test scenario**: 
  1. Start server: `docker compose up` or local setup
  2. Open `client/screen_wss.html` in browser
  3. Connect to server using WebSocket URL
  4. Select screen sharing and start streaming
  5. Test at least one processor (Basic Processor ID: 0 is fastest)
  6. Verify processor returns results and no errors in server logs
- **Audio testing**: Test "Start Audio" button for voice processor selection
- **Processor testing**: Verify enabled processors (0, 2, 4, 11) work correctly

### Build Validation
- **ALWAYS** run environment setup validation: `conda run -n whatsai python -c "import fastapi, uvicorn, torch, transformers; print('Core dependencies working')"`
- **Module loading test**: `conda run -n whatsai python -c "import processors.basic_processor; print('Processors module working')"`
- **Configuration validation**: Check `processor_config.json` for enabled processors
- **API endpoint test**: Start server and test `curl http://localhost:8000/processors` returns JSON with enabled processors

## Timing Expectations

### Build Times (NEVER CANCEL)
- **Conda environment creation**: 4-6 minutes per environment
- **Docker build**: 15-25 minutes (includes model downloads)
- **Docker compose up**: 2-3 minutes for server startup
- **Processor startup**: 1-2 minutes for all enabled processors to initialize

### Timeout Recommendations
- **Environment creation**: 10+ minutes timeout
- **Docker build**: 30+ minutes timeout  
- **Server startup**: 5+ minutes timeout
- **End-to-end testing**: 10+ minutes timeout

## Common Tasks

### Repository Structure
```
HackTemplate/
├── README.md                    # Main documentation
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # Container build definition
├── processor_config.json       # Processor configuration
├── stream_wss.py               # Main WebSocket server
├── stream_sonic.py             # Alternative server
├── stream_rtc.py               # RTC variant server
├── processors/                 # AI processor modules
├── audio_processors/           # Audio processing modules
├── client/                     # Web client files
├── resources/                  # Build and environment scripts
└── models/                     # ML model storage
```

### Key Configuration Files
- **processor_config.json**: Defines available processors, ports, and conda environments
- **.env**: Required file with `GEMINI_API_KEY="your-api-key"`
- **resources/whatsai/pyproject.toml**: Main AI environment dependencies
- **resources/aws/pyproject.toml**: AWS integration dependencies

### Enabled Processors (Default)
- **ID 0**: Basic Processor (pass-through, testing)
- **ID 2**: Scene Captioning Processor (Florence-2 OCR/captioning)  
- **ID 4**: Scene Object Processor (YOLO11 object detection)
- **ID 11**: Card Processor (MTG card recognition)

### Disabled Processors (Default)
- **ID 1**: Depth Processor (MediaPipe finger counting)
- **ID 3**: Region Processor (CamIO object recognition)
- **ID 5-10**: Various panel processors (switch, thermostat, elevator, etc.)

## Dependencies and Models

### Core Python Dependencies
- **whatsai environment**: torch, transformers, ultralytics, mediapipe, fastapi, uvicorn, google-genai
- **aws environment**: google-genai, aws-sdk-bedrock-runtime, httpx
- **Additional required**: google-genai must be manually installed in whatsai environment

### Downloaded Models (Automatic)
- **YOLO11**: `yolo11n-seg.pt` (object detection/segmentation)
- **Florence-2**: `microsoft/Florence-2-large` (scene captioning)
- **Hierarchical-Localization**: Computer vision localization tools

## Troubleshooting

### Docker Build Issues
- **SSL certificate errors**: Common in CI environments, use local conda setup instead
- **Network timeouts**: Increase Docker build timeout, check internet connectivity
- **CUDA compatibility**: Project requires NVIDIA GPU support for optimal performance

### Runtime Issues  
- **Port conflicts**: Check if ports 8000-8012 are available
- **Conda activation**: Ensure conda is properly initialized: `conda init bash`
- **Missing models**: Models download automatically on first processor use
- **WebSocket connection**: Verify server URL format includes `/ws` endpoint

### Performance Optimization
- **GPU acceleration**: Project optimized for CUDA-enabled environments
- **Memory usage**: Large ML models require 8GB+ RAM
- **Network bandwidth**: Screen streaming requires stable connection

## Development Guidelines

### Making Changes
- **ALWAYS** test with Basic Processor (ID: 0) first for connectivity
- **Processor changes**: Modify files in `processors/` directory
- **Server changes**: Main logic in `stream_wss.py`, `stream_sonic.py`
- **Client changes**: Modify `client/screen_wss.html`
- **Configuration**: Update `processor_config.json` for new processors

### Testing Protocol
1. **Environment validation**: Verify conda environments work
2. **Build validation**: Ensure Docker build succeeds  
3. **Server startup**: Test server starts without errors
4. **Client connection**: Verify WebSocket connectivity
5. **Processor testing**: Test at least one enabled processor
6. **End-to-end flow**: Complete screen sharing workflow

### Common Command Patterns
- **Environment activation**: `conda activate whatsai` or `conda activate aws`
- **Server startup**: `uvicorn stream_wss:app --host 0.0.0.0 --port 8000`
- **Docker operations**: `docker compose build` → `docker compose up`
- **Testing**: Open browser → navigate to `client/screen_wss.html` → test connectivity

## Known Limitations

### Build Environment Restrictions
- **Network dependencies**: Requires internet for model downloads
- **SSL certificates**: May fail in containerized CI environments
- **GPU requirements**: Optimal performance requires NVIDIA GPU
- **Platform support**: Designed for Linux/WSL2 environments

### Runtime Constraints
- **Browser compatibility**: Requires modern browser with WebRTC support
- **Permissions**: Screen sharing requires browser permission grants
- **API keys**: Gemini API key required for audio processing features
- **Resource usage**: High memory/GPU usage during active processing

**CRITICAL REMINDER**: NEVER CANCEL long-running operations. Builds and environment setups are time-intensive but essential for proper functionality.