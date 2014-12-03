
#!/bin/bash
set -e

PREFIX="$HOME/local"
DOWNLOAD_DIR="/tmp/keckj/download"
LOG="/tmp/keckj/log_build.log"

ARGS=""
THREADS=8

BZIP2="bzip2-1.0.6.tar.gz"
XZ="xz-5.0.7.tar.gz"
LZIP="lzip-1.16.tar.gz"
CMAKE="cmake-3.0.2.tar.gz"

BZIP2_URL="http://www.bzip.org/1.0.6/$BZIP2"
XZ_URL="http://tukaani.org/xz/$XZ"
LZIP_URL="http://download.savannah.gnu.org/releases/lzip/$LZIP"
CMAKE_URL="http://www.cmake.org/files/v3.0/$CMAKE"

source "$(pwd)/utils.sh"

mkdir -p "$DOWNLOAD_DIR"
rm -f "$LOG"
#downloadExtractAndBuild "$DOWNLOAD_DIR" "$BZIP2_URL" "$BZIP2" "$PREFIX" "$ARGS"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$XZ_URL" "$XZ" "$PREFIX" "$ARGS"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$LZIP_URL" "$LZIP" "$PREFIX" "$ARGS"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$CMAKE_URL" "$CMAKE" "$PREFIX" "$ARGS"

echo "ALL DONE !"
