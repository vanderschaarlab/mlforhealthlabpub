echo "running data.py"
python3 diag-bias/data.py --bias

echo "running main-interpole.py"
python3 diag-bias/main-interpole.py --bias
echo "running main-offpoirl.py"
python3 diag-bias/main-offpoirl.py --bias
echo "running main-poirl.py"
python3 diag-bias/main-poirl.py --bias
echo "running main-pombil.py"
python3 diag-bias/main-pombil.py --bias
echo "running main-rbc.py"
python3 diag-bias/main-rbc.py --bias

echo "running eval.py"
python3 diag-bias/eval.py --bias > res/results_bias.txt
