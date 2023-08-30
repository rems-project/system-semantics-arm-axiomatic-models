#!/bin/sh
$(dirname "$0")/_runner.py nightly $@ --ifetch --extraargs="--pc-limit=3" --extraargs="--pc-limit-mode=discard"
