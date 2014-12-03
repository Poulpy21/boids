#!/bin/bash
set -e
source "$(pwd)/utils.sh"

LOG="$HOME/log_build.log"
rm -f "$LOG"

GMP="gmp-6.0.0a.tar.lz"
MPFR="mpfr-3.1.2.tar.xz"
MPC="mpc-1.0.1.tar.gz"
ISL="isl-0.12.tar.bz2"
CLOOG="cloog-0.18.1.tar.gz"
LIB_ELF="libelf-0.8.13.tar.gz"
PPL="ppl-1.1.tar.xz"
GCC="gcc-4.8.2.tar.bz2"

GMP_URL="https://gmplib.org/download/gmp/$GMP"
MPFR_URL="http://www.mpfr.org/mpfr-current/$MPFR"
MPC_URL="ftp://ftp.gnu.org/gnu/mpc/$MPC"
ISL_URL="http://isl.gforge.inria.fr/$ISL"
CLOOG_URL="http://www.bastoul.net/cloog/pages/download/$CLOOG"
LIB_ELF_URL="http://www.mr511.de/software/$LIB_ELF"
PPL_URL="ftp://ftp.cs.unipr.it/pub/ppl/releases/1.1/$PPL"
GCC_URL="http://www.netgull.com/gcc/releases/$(extractBasename $GCC)/$GCC"

TMP_PREFIX="/tmp/gcc_keckj/install"
GCC_PREFIX="$HOME/gcc-4.8.2"
DOWNLOAD_DIR="/tmp/gcc_keckj/download"
GCC_BUILD_DIR="/tmp/gcc_keckj/build"

ARGS="--disable-shared --enable-static"
ARGS_WITH_GMP="--disable-shared --enable-static --with-gmp=${TMP_PREFIX}"
ARGS_WITH_GMP_PREFIX="--disable-shared --enable-static --with-gmp-prefix=${TMP_PREFIX}"
GCC_ARGS="--prefix=${GCC_PREFIX} --enable-bootstrap --disable-shared --enable-static --disable-multilib --enable-languages=c,c++ --with-mpc-include=${TMP_PREFIX}/include --with-mpc-lib=${TMP_PREFIX}/lib --with-mpfr-include=${TMP_PREFIX}/include --with-mpfr-lib=${TMP_PREFIX}/lib --with-gmp-include=${TMP_PREFIX}/include --with-gmp-lib=${TMP_PREFIX}/lib --with-cloog-include=${TMP_PREFIX}/include/cloog --with-cloog-lib=${TMP_PREFIX}/lib --with-isl-include=${TMP_PREFIX}/include --with-isl-lib=${TMP_PREFIX}/lib --with-libelf-include=${TMP_PREFIX}/include --with-libelf-lib=${TMP_PREFIX}/lib --with-ppl-include=${TMP_PREFIX} --with-ppl-lib=${TMP_PREFIX}"

THREADS=8

mkdir -p "$DOWNLOAD_DIR"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$GMP_URL" "$GMP" "$TMP_PREFIX" "$ARGS --enable-cxx"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$PPL_URL" "$PPL" "$TMP_PREFIX" "$ARGS_WITH_GMP"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$MPFR_URL" "$MPFR" "$TMP_PREFIX" "$ARGS_WITH_GMP"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$MPC_URL" "$MPC" "$TMP_PREFIX" "$ARGS_WITH_GMP"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$CLOOG_URL" "$CLOOG" "$TMP_PREFIX" "$ARGS_WITH_GMP_PREFIX"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$LIB_ELF_URL" "$LIB_ELF" "$TMP_PREFIX" "$ARGS_WITH_GMP"
downloadExtractAndBuild "$DOWNLOAD_DIR" "$ISL_URL" "$ISL" "$TMP_PREFIX" "$ARGS_WITH_GMP_PREFIX"

cd "$DOWNLOAD_DIR"
if [ ! -f "$DOWNLOAD_DIR/$GCC" ]; then
    echo $GCC
    wget "$GCC_URL"
fi

dir="$DOWNLOAD_DIR/$(extractBasename $GCC)"
if [ ! -d "$dir" ]; then
    echo "Creating directory $dir ..."
    mkdir -p "$dir"
    extract "$DOWNLOAD_DIR/$GCC" "$dir"
fi

rm -Rf "$GCC_PREFIX"
rm -Rf "$GCC_BUILD_DIR"
mkdir -p "$GCC_PREFIX"
mkdir -p "$GCC_BUILD_DIR"
cd "$GCC_BUILD_DIR"

echo "BUILDING $GCC ..."
echo "CONFIGURE ($GCC) $GCC_ARGS"
$dir/configure $GCC_ARGS
echo "MAKE ($GCC)"
make "-j$THREADS"
echo "MAKE INSTALL ($GCC)"
make install
echo "BUILD DONE ($GCC)"

echo "ALL DONE !"

