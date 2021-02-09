# Prepare a "working directory" at $COMP_ROOT/WKG, needed for locally debug the competition backend code.
# Note: the /opt/hide-and-seek/ directory needs to *also* be prepared.

# Set:
REPO_DIR="/home/ubuntu/mlforhealthlabcode"

# ----------------------------------------------------------------------------------------------------------------------

COMP_ROOT="$REPO_DIR/app/hide-and-seek"

[ ! -d $COMP_ROOT/WKG ]             && mkdir $COMP_ROOT/WKG
[ ! -d $COMP_ROOT/WKG/RUN ]         && mkdir $COMP_ROOT/WKG/RUN
[ ! -d $COMP_ROOT/WKG/RUN/input ]   && mkdir $COMP_ROOT/WKG/RUN/input
[ ! -d $COMP_ROOT/WKG/RUN/output ]  && mkdir $COMP_ROOT/WKG/RUN/output
[ ! -d $COMP_ROOT/WKG/RUN/program ] && mkdir $COMP_ROOT/WKG/RUN/program
[ ! -d $COMP_ROOT/WKG/RUN/temp ]    && mkdir $COMP_ROOT/WKG/RUN/temp
[ ! -d $COMP_ROOT/WKG/SHARED ]      && mkdir $COMP_ROOT/WKG/SHARED

cd $COMP_ROOT/WKG/RUN

# 1. program/
echo "PREPARING: program/"
sudo rm -rf ./program/*
cp $COMP_ROOT/competition/scoring_program/metadata   ./program/metadata
cp $COMP_ROOT/competition/scoring_program/scoring.py ./program/scoring.py
cp $COMP_ROOT/competition/scoring_program/running.py ./program/running.py
cp -r -L $COMP_ROOT/competition/scoring_program/data        ./program/data
cp -r -L $COMP_ROOT/competition/scoring_program/metrics     ./program/metrics
cp -r -L $COMP_ROOT/competition/scoring_program/computils   ./program/computils

# 2. input/
# 2.1. input/hider_add_noise/
echo "PREPARING: input/hider_add_noise/"
rm -rf ./input/hider_add_noise/
mkdir -p ./input/hider_add_noise/res
rm -rf ./input/temp/
cp -r -L $COMP_ROOT/competition/baselines/hider_add_noise/ ./input/temp/
mv ./input/temp/** ./input/hider_add_noise/res/
echo "submitted-by: DEBUG_hider_add_noise" > ./input/hider_add_noise/metadata
rm -rf ./input/temp/

# 2.2. input/hider_timegan/
echo "PREPARING: input/hider_timegan/"
rm -rf ./input/hider_timegan/
mkdir -p ./input/hider_timegan/res
rm -rf ./input/temp/
cp -r -L $COMP_ROOT/competition/baselines/hider_timegan/ ./input/temp/
mv ./input/temp/** ./input/hider_timegan/res/
echo "submitted-by: DEBUG_hider_timegan" > ./input/hider_timegan/metadata
rm -rf ./input/temp/

# 2.3. input/seeker_binary_predictor/
echo "PREPARING: input/seeker_binary_predictor/"
rm -rf ./input/seeker_binary_predictor/
mkdir -p ./input/seeker_binary_predictor/res
rm -rf ./input/temp/
cp -r -L $COMP_ROOT/competition/baselines/seeker_binary_predictor/ ./input/temp/
mv ./input/temp/** ./input/seeker_binary_predictor/res/
echo "submitted-by: DEBUG_seeker_binary_predictor" > ./input/seeker_binary_predictor/metadata
rm -rf ./input/temp/

# 2.4. input/seeker_knn/
echo "PREPARING: input/seeker_knn/"
rm -rf ./input/seeker_knn/
mkdir -p ./input/seeker_knn/res
rm -rf ./input/temp/
cp -r -L $COMP_ROOT/competition/baselines/seeker_knn/ ./input/temp/
mv ./input/temp/** ./input/seeker_knn/res/
echo "submitted-by: DEBUG_seeker_knn" > ./input/seeker_knn/metadata
rm -rf ./input/temp/

echo "DONE"