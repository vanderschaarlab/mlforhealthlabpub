# uncomment if processed data files are missing
# echo "running data0.py ..."
# python3 adni/data0.py

echo "running data.py ..."
python3 adni/data.py

echo "running main.py ..."
python3 adni/main.py
echo "running main.py for age ..."
python3 adni/main.py --tag age
echo "running main.py for apoe ..."
python3 adni/main.py --tag apoe
echo "running main.py for female ..."
python3 adni/main.py --tag female
