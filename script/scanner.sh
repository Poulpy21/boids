#! /bin/bash

set -e

min_ensipc_id=10
max_ensipc_id=338
ensipc_count=2

min_ensipsi_id=1
max_ensipsi_id=100
ensipsi_count=3

exec 3>&2
exec 4>&1

function getOnlineHosts() { #$1 = prefix $2 = min_id $3 = max_id $4 = count
    local hosts=$(printf "$1%d " $(seq $2 $3))
    local online_hosts=$(nmap -sP $hosts 2>/dev/null | grep "Nmap scan" | awk -F" " '{print $5}')
    echo "$(echo $online_hosts | wc -l) $1s are online waiting for you!" >&3
    echo $(echo $online_hosts | shuf | tail "-$4") >&4
}

getOnlineHosts "ensipc" "$min_ensipc_id" "$max_ensipc_id" "$ensipc_count"
getOnlineHosts "ensipsi" "$min_ensipsi_id" "$max_ensipsi_id" "$ensipsi_count"

exit 0
