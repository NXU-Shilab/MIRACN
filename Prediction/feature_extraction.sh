columns=$(cat ./feature.txt)
cut -f "$columns" "$1" > "$2"
