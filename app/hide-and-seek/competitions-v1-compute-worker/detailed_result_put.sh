# $1 -> the URL to write to
# $2 -> the file to watch

while sleep 1 ; do
    # -d makes entr watch for non-existing files as well
    echo $2 | entr -d python /worker/detailed_result_put.py $1 $2
done
