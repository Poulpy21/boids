#! /bin/bash

set -e

min_ensipc_id=1
#max_ensipc_id=338
max_ensipc_id=40
ensipc_count=2

min_ensipsys_id=1
max_ensipsys_id=100
ensipsys_count=3

exec 3>&2
exec 4>&1

function getOnlineHosts() { #$1 = prefix $2 = min_id $3 = max_id $4 = count
    local hosts=$(printf "$1%d " $(seq $2 $3))
    local online_hosts=$(nmap -sP $hosts 2>/dev/null | grep "Nmap scan" | awk -F" " '{print $5}')
    #echo "$(echo $online_hosts | wc -w) $1s are online waiting for you!" >&3
    #echo $(echo $online_hosts | xargs -n1 | shuf | xargs | cut -d " " "-f-$4") >&4
    echo $(echo $online_hosts | cut -d " " "-f-$4") >&4
}

getOnlineHosts "ensipc" "$min_ensipc_id" "$max_ensipc_id" "$ensipc_count"
#getOnlineHosts "ensipsys" "$min_ensipsys_id" "$max_ensipsys_id" "$ensipsys_count"

exit 0
