# uncomment if processed data files are missing
# echo "running data.py"
# python3 adni/data.py

for i in {0..4}
do
    echo "i: $i, alg: interpole"
    python3 adni/main-interpole.py -i $i --silent
    echo "i: $i, alg: offpoirl"
    python3 adni/main-offpoirl.py -i $i --silent
    echo "i: $i, alg: poirl"
    python3 adni/main-poirl.py -i $i --silent
    echo "i: $i, alg: pombil"
    python3 adni/main-pombil.py -i $i --silent
    echo "i: $i, alg: rbc"
    python3 adni/main-rbc.py -i $i --silent
done

echo "running eval.py"
python3 adni/eval.py > res/adni/results_adni.txt
