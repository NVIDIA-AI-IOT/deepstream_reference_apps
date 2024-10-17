
make

## enable NVSTREAMMUX_ADAPTIVE_BATCHING
if [ x"$NVSTREAMMUX_ADAPTIVE_BATCHING" != x"yes" ]; then
  export NVSTREAMMUX_ADAPTIVE_BATCHING=yes
  rm -rf ~/.cache/gstreamer-1.0/
  echo "export NVSTREAMMUX_ADAPTIVE_BATCHING=yes"
fi

## dict.txt is label file for LPR model
if [ ! -f dict.txt ]; then
  wget 'https://api.ngc.nvidia.com/v2/models/nvidia/tao/lprnet/versions/deployable_v1.0/files/us_lp_characters.txt' -O dict.txt
fi
