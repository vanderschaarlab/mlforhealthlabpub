echo "running data.py ..."
python3 diag/data.py

echo "running main.py for alp-lowlow ..."
python3 diag/main.py --tag alp-lowlow
echo "running main.py for alp-low ..."
python3 diag/main.py --tag alp-low
echo "running main.py for alp-med ..."
python3 diag/main.py --tag alp-med
echo "running main.py for alp-hgh ..."
python3 diag/main.py --tag alp-hgh
echo "running main.py for bet ..."
python3 diag/main.py --tag bet
echo "running main.py for eta ..."
python3 diag/main.py --tag eta

echo "running main-irl.py for bet ..."
python3 diag/main-irl.py --tag bet
echo "running main-ril.py for eta ..."
python3 diag/main-irl.py --tag eta
