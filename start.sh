#!/bin/bash

sudo -E LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH HF_HOME=$HOME/.cache/huggingface venv/bin/python whisper_ptt_cuda.py




