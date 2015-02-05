#! /bin/bash

# Usage: ./scanner.sh | ./genAppFile.sh login

set -e

function generateAppFile() { #$1 = list of hosts $2 = file name $3 = username
    > $2
    for host in $1
    do
        echo "-host "$3"@"$host "-np 1 main" >> $2
    done
}
script_path=$(dirname `which $0`)
appfile_path="$script_path/../config/appfile"

read stdin;
generateAppFile "$stdin" $appfile_path $1
