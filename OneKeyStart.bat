@echo off

:: Set TorchAudio environment variable to suppress warnings
set TORCHAUDIO_USE_BACKEND_DISPATCHER=1

:: GPU Detection and Environment Setup
echo Detecting GPU type...
nvidia-smi --query-gpu=name --format=csv,noheader,nounits > gpu_info.tmp 2>nul
if exist gpu_info.tmp (
    findstr /i "RTX 50" gpu_info.tmp >nul
    if not errorlevel 1 (
        echo RTX 50 Series GPU detected, setting compatibility variables...
        set TORCH_CUDA_ARCH_LIST=7.0 7.5 8.0 8.6 8.9 9.0+PTX
        set NVIDIA_ALLOW_UNSUPPORTED_ARCHS=true
        echo Environment variables set successfully
    )
    del gpu_info.tmp
)

call conda activate videolingo
python -m streamlit run st.py
pause
