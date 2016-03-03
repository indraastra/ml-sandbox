#!/usr/bin/env bash

# 1. Download all Google fonts.
if [ ! -f tmp/fonts.zip ]
then
    mkdir tmp
    wget https://github.com/google/fonts/archive/master.zip -o tmp/fonts.zip
fi

# 2. Extract only TrueType fonts into current directory without preserving
# subdirectory structure and without overwriting files.
unzip -j -n tmp/fonts.zip "*.ttf" -d fonts/
