function colorecho(){
   
    exec 1>&4

    local exp=$1;
    local color=$2;
    if ! [[ $color =~ '^[0-9]$' ]] ; then
       case $(echo $color | tr '[:upper:]' '[:lower:]') in
        black) color=0 ;;
        red) color=1 ;;
        green) color=2 ;;
        yellow) color=3 ;;
        blue) color=4 ;;
        magenta) color=5 ;;
        cyan) color=6 ;;
        white|*) color=7 ;; # white or invalid color
       esac
    fi
    tput setaf $color;
    echo $exp;
    tput sgr0;
    
    exec 1>&3
}

function scriptFolder() {
    local SOURCE="${BASH_SOURCE[0]}"
    
    # resolve $SOURCE until the file is no longer a symlink
    while [ -h "$SOURCE" ]; do
        local DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        local SOURCE="$(readlink "$SOURCE")"
        # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" 
    done
    
    local DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    echo "$DIR"
}


function extract() {
    #echo "Extracting $1 to folder $2 ..."
    
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
        echo "'$1' is not a file!"
        exit 2
    fi
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



function build() { #$1=dir #2=prefix #3=additional args
    colorecho "BUILDING $1 ..." blue
    colorecho "CONFIGURE ($1) $3 --prefix=$2" yellow
    mkdir -p "$2"
    mkdir -p "$1/build"
    cd "$1/build"
    ../configure $3 "--prefix=$2"
    colorecho "MAKE ($1)" yellow
    make "-j$THREADS"
    #colorecho "MAKE CHECK ($1)" yellow
    #make check "-j$THREADS"
    colorecho "MAKE INSTALL ($1)" yellow
    make install
    colorecho "BUILD OK ($1)" blue
    echo ""
}

function downloadExtractAndBuild() { #folder #url #filename #prefix #additionalargs
    cd "$1"
    if [ ! -f "$1/$3" ]; then
        colorecho "DOWNLOADING ($3)..." yellow
        wget "$2"
    fi

    local dir="$1/$(extractBasename $3)"
    if [ ! -d "$dir" ]; then
        colorecho "EXTRACTING ($3)..." yellow
        mkdir -p "$dir"
        extract "$1/$3" "$dir"
    fi

    build "$dir" "$4" "$5"
}

function install() { #$1=dl_dir $2=url $3=name $4=prefix $5=args $6=successlogfile
    if [ -z "$(cat $6 | grep -o $3)" ]; then
        downloadExtractAndBuild "$1" "$2" "$3" "$4" "$5"
        echo "$3" >> "$6"
    fi 
    
    colorecho "$(extractBasename $3) ... ok" green
}


function failed() {
    local errcode=$?
    local command=${BASH_COMMAND}
    local line=${BASH_LINENO[0]}

    echo "" >&4
    if [ $errcode -eq 0 ]; then
        colorecho "All good, bye !" green
    else
        colorecho "### FAILURE ###" red
        colorecho "Error code ${errcode} in command ${command} on line ${line}!" red 
        colorecho "###############" red
    fi
    echo "" >&4

    exec 1>&-
    exec 4>&-
    exec 2>&-
    exec 3>&-
}
