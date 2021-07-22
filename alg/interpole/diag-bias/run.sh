echo "running data.py"
python3 diag-bias/data.py

echo "running main-interpole.py"
python3 diag-bias/main-interpole.py
echo "running main-offpoirl.py"
python3 diag-bias/main-offpoirl.py
echo "running main-poirl.py"
python3 diag-bias/main-poirl.py
echo "running main-pombil.py"
python3 diag-bias/main-pombil.py
echo "running main-rbc.py"
python3 diag-bias/main-rbc.py

echo "running eval.py"
python3 diag-bias/eval.py > res/results_diag.txt
