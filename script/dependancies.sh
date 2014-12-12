#!/bin/bash

set -e

if [ $# -ne 1 ]; then
    RESET=0
else
    RESET="$1"
fi

PREFIX="${HOME}/local"
TMP_DIR="/tmp/keckj"

DOWNLOAD_DIR="${TMP_DIR}/download"
LOG="${TMP_DIR}/log_build.log"
SUCCESS="${TMP_DIR}/success.log"

ARGS=""
ARGS_WITH_GMP="$ARGS --with-gmp=${PREFIX}"
ARGS_WITH_GMP_PREFIX="$ARGS --with-gmp-prefix=${PREFIX}"
ARGS_WITH_LIBGMP_PREFIX="$ARGS --with-libgmp-prefix=${PREFIX}"

THREADS=8

source "$(pwd)/utils.sh"
trap failed ERR EXIT HUP INT QUIT

rm -f "$LOG"
if [ ${RESET} -eq 2 ]; then
    echo "Reseting download folder"
    rm -Rf ${DOWNLOAD_DIR}
    rm ${SUCCESS}
elif [ ${RESET} -eq 1 ]; then
    echo "Reseting dowload dir extracted folders (keep downloads)"
    rm -Rf $(find ${DOWNLOAD_DIR} -mindepth 1 -maxdepth 1 -type d)
    rm ${SUCCESS}
fi

mkdir -p ${TMP_DIR}
mkdir -p "${DOWNLOAD_DIR}"

touch "${LOG}"
touch "${SUCCESS}"

exec 3> ${LOG}
exec 4> /dev/stdout
exec 1>&3
exec 2> >(tee "${LOG}.error" >&2)

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig"
export PATH="${PREFIX}/bin:${PATH}"
export LD_LIBRARYPATH="${PREFIX}/lib:${PREFIX}/lib64:${LD_LIBRARYPATH}"

BINUTILS="binutils-2.24.tar.gz"
BINUTILS_URL="http://ftp.gnu.org/gnu/binutils/$BINUTILS"
install "${DOWNLOAD_DIR}" "${BINUTILS_URL}" "${BINUTILS}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

LIBC="glibc-2.20.tar.gz"
LIBC_URL="https://ftp.gnu.org/gnu/libc/$LIBC"
install "${DOWNLOAD_DIR}" "${LIBC_URL}" "${LIBC}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

GMP="gmp-6.0.0a.tar.lz"
GMP_URL="https://gmplib.org/download/gmp/$GMP"
install "${DOWNLOAD_DIR}" "${GMP_URL}" "${GMP}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#UNISTRING="libunistring-0.9.4.tar.gz"
#UNISTRING_URL="http://ftp.gnu.org/gnu/libunistring/$UNISTRING"
#install "${DOWNLOAD_DIR}" "${UNISTRING_URL}" "${UNISTRING}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#FFI="libffi-3.2.1.tar.gz"
#FFI_URL="ftp://sourceware.org/pub/libffi/$FFI"
#install "${DOWNLOAD_DIR}" "${FFI_URL}" "${FFI}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#GC="gc-7.2f.tar.gz"
#GC_URL="http://www.hboehm.info/gc/gc_source/$GC"
#install "${DOWNLOAD_DIR}" "${GC_URL}" "${GC}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#GNU_READLINE="readline-6.3.tar.gz"
#GNU_READLINE_URL="https://ftp.gnu.org/gnu/readline/$GNU_READLINE"
#install "${DOWNLOAD_DIR}" "${GNU_READLINE_URL}" "${GNU_READLINE}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#GUILE="guile-2.0.11.tar.gz"
#GUILE_URL="ftp://ftp.gnu.org/gnu/guile/$GUILE"
#install "${DOWNLOAD_DIR}" "${GUILE_URL}" "${GUILE}" "${PREFIX}" "${ARGS_WITH_LIBGMP_PREFIX}" "${SUCCESS}"

#FLEX="flex-2.5.39.tar.gz"
#FLEX_URL="http://sourceforge.net/projects/flex/files/$FLEX/download"
#install "${DOWNLOAD_DIR}" "${FLEX_URL}" "${FLEX}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

#BISON="bison-2.7.91.tar.gz"
#BISON_URL="ftp://alpha.gnu.org/gnu/bison/$BISON"
#install "${DOWNLOAD_DIR}" "${BISON_URL}" "${BISON}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

M4="m4-latest.tar.gz"
M4_URL="http://ftp.gnu.org/gnu/m4/$M4"
install "${DOWNLOAD_DIR}" "${M4_URL}" "${M4}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

AUTOCONF="autoconf-latest.tar.gz"
AUTOCONF_URL="http://ftp.gnu.org/gnu/autoconf/$AUTOCONF"
install "${DOWNLOAD_DIR}" "${AUTOCONF_URL}" "${AUTOCONF}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

AUTOMAKE="automake-1.14.1.tar.gz"
AUTOMAKE_URL="http://ftp.gnu.org/gnu/automake/$AUTOMAKE"
install "${DOWNLOAD_DIR}" "${AUTOMAKE_URL}" "${AUTOMAKE}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

AUTOGEN="autogen-5.18.4.tar.gz"
AUTOGEN_URL="http://mirrors.ibiblio.org/gnu/ftp/gnu/autogen/rel5.18.4/$AUTOGEN"
install "${DOWNLOAD_DIR}" "${AUTOGEN_URL}" "${AUTOGEN}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

MAKE="make-4.1.tar.gz"
MAKE_URL="https://ftp.gnu.org/gnu/make/$MAKE"
install "${DOWNLOAD_DIR}" "${MAKE_URL}" "${MAKE}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

BZIP2="bzip2-1.0.6.tar.gz"
BZIP2_URL="http://www.bzip.org/1.0.6/$BZIP2"
install "${DOWNLOAD_DIR}" "${BZIP2_URL}" "${BZIP2}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

XZ="xz-5.0.7.tar.gz"
XZ_URL="http://tukaani.org/xz/$XZ"
install "${DOWNLOAD_DIR}" "${XZ_URL}" "${XZ}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

LZIP="lzip-1.16.tar.gz"
LZIP_URL="http://download.savannah.gnu.org/releases/lzip/$LZIP"
install "${DOWNLOAD_DIR}" "${LZIP_URL}" "${LZIP}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

CMAKE="cmake-3.0.2.tar.gz"
CMAKE_URL="http://www.cmake.org/files/v3.0/$CMAKE"
install "${DOWNLOAD_DIR}" "${CMAKE_URL}" "${CMAKE}" "${PREFIX}" "${ARGS}" "${SUCCESS}"

exit 0
