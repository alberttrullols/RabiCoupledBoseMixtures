#!/bin/bash
julia -t auto ../run_dmc_cc_forward_correfoc.jl 2>&1 | tee run.log
