#!/bin/bash
# stage_egamma_25.sh — copy all files in a fileset JSON to EOS via xrootd door

FILESET="fileset_all_25_Z_reduced.json"          # <-- your fileset json
DEST="root://cmseos.fnal.gov//store/user/lpcsusylep/temp_egamma_files_25"
RETRIES=3

# pull the file paths (JSON keys under each dataset's "files")
files=$(jq -r '.[].files | keys[]' "$FILESET")

n=$(echo "$files" | wc -l)
echo "Found $n files to stage -> $DEST"
echo

fail=0
i=0
for src in $files; do
    i=$((i+1))
    fname=$(basename "$src")

    # skip if already on EOS (xrdfs stat succeeds -> exists)
    if xrdfs cmseos.fnal.gov stat "/store/user/lpcsusylep/temp_egamma_files_25/$fname" &>/dev/null; then
        echo "[$i/$n] SKIP (exists): $fname"
        continue
    fi

    ok=0
    for attempt in $(seq 1 $RETRIES); do
        echo "[$i/$n] copy (try $attempt): $fname"
        if xrdcp --silent --retry 2 "$src" "$DEST/"; then
            ok=1
            break
        fi
        echo "    ... failed, retrying"
        sleep 5
    done

    if [ $ok -eq 0 ]; then
        echo "[$i/$n] GIVE UP: $src"
        echo "$src" >> failed_copies.txt
        fail=$((fail+1))
    fi
done

echo
echo "Done. $((n-fail))/$n succeeded."
[ $fail -gt 0 ] && echo "Failures logged to failed_copies.txt"