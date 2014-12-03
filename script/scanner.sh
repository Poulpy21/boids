#! /bin/bash

hosts=""
for i in {10..338}; do
    hosts="$hosts ensipc$i"
done
for i in {1..100}; do
    hosts="$hosts ensipsys$i"
done
#echo $hosts
#nmap -sP -iL hostfile 2>/dev/null | grep "Nmap scan" | awk -F" " '{print $5}'
nmap -sP $hosts 2>/dev/null | grep "Nmap scan" | awk -F" " '{print $5}' > onlineHosts
