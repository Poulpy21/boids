set -e

function extract() {
echo "Extracting $1 to folder $2 ..."
if [ -f $1 ]; then
    case $1 in
        *.tar.bz2)   tar xvjf $1 -C "$2" --strip-components=1;;
        *.tar.gz)    tar xvzf $1 -C "$2" --strip-components=1;;
        *.tar.xz)    tar xvJf $1 -C "$2" --strip-components=1;;
        *.tar.lz)    tar --lzip -xvf $1 -C "$2" --strip-components=1;;
        *.bz2)       bunzip2 $1 -C "$2" --strip-components=1;;
        *.gz)        gunzip $1 -C "$2" --strip-components=1;;
        *.tar)       tar xvf $1 -C "$2" --strip-components=1;;
        *.tgz)       tar xvzf $1 -C "$2" --strip-components=1;;
        *)           echo "I don't know how to extract '$1'..."; exit 1; ;;
    esac
else
    echo "'$1' is not a valid file!"
    exit 1
fi
}

function build() { #$1=dir #2=prefix #3=additional args
    echo "BUILDING $1 ..."
    echo "CONFIGURE ($1) $3 --prefix=$2"
    cd "$1"
    mkdir -p "$2"
    ./configure $3 "--prefix=$2" >> "$LOG"
    echo "MAKE ($1)"
    make "-j$THREADS" >> "$LOG"
    #echo "MAKE CHECK ($1)"
    #make check "-j$THREADS" >> "$LOG"
    echo "MAKE INSTALL ($1)"
    make install >> "$LOG"
    echo "BUILD OK ($1)"
    echo ""
}

function downloadExtractAndBuild() { #folder #url #filename #prefix #additionalargs

    cd "$1"
    if [ ! -f "$1/$3" ]; then
        wget "$2"
    fi

    local dir="$1/$(extractBasename $3)"
    if [ ! -d "$dir" ]; then
        echo "Creating directory $dir ..."
        mkdir -p "$dir"
        extract "$1/$3" "$dir"
    fi

    build "$dir" "$4" "$5"
}


function extractBasename() {
    local ext
    case $1 in
        *.tar.bz2)   ext=".tar.bz2"   ;;
        *.tar.gz)    ext=".tar.gz"    ;;
        *.tar.xz)    ext=".tar.xz"    ;;
        *.tar.lz)    ext=".tar.lz"    ;;
        *.bz2)       ext=".bz2"       ;;
        *.gz)        ext=".gz"        ;;
        *.tar)       ext=".tar"       ;;
        *.tgz)       ext=".tgz"       ;;
        *)           echo "I don't know how to extract '$1'..."; exit 1; ;;
    esac

    echo $(basename $1 $ext)
}
