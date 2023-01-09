#!/bin/sh

echo "COMPILING MODEL FOR KV260.."

ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
TARGET=kv260

vai_c_xir \
  --xmodel      build/quant_model/Net_int.xmodel \
  --arch        $ARCH \
  --net_name    group13_${TARGET} \
  --output_dir  build/compiled_model

echo "MODEL COMPILED"
