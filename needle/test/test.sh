#!/usr/bin/env sh
# -*- coding: utf-8 -*-

export NEEDLE_BACKEND=np
pytest

export NEEDLE_BACKEND=nd
pytest
