#!/bin/bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install -r manga_colorization_v2/requirements.txt