echo_log() {
    echo "$1" >> $LOG
    echo "$1"
}

time_init() {
    TIME_START=$(date +%s)
}


time_log() {
    TIME_STOP=$(date +%s)
    TIME_DURATION=$(( $TIME_STOP - $TIME_START ))
    echo_log "time=$TIME_DURATION $1"
    TIME_START=$(date +%s)
}

set -e
set -o pipefail

# x=['Normal','AR_Normal','Uniform','AR_Uniform']
# y=['Logit','Gauss']

if [[ ! "$XNAME_LST" ]]; then
    XNAME_LST="AR_Normal"
fi
if [[ ! "$YNAME_LST" ]]; then
    YNAME_LST="Gauss Logit"
fi

TMPROOT=$HOME/tmp/gen
PYTHONDEFAULT=python3
VERSION=1

for XNAME in $XNAME_LST
do
    for YNAME in $YNAME_LST
    do
        IDIR="$TMPROOT/knockoff_data/v_${VERSION}/xname_${XNAME}/yname_${YNAME}"
        RESDIR="result/v_${VERSION}/h_${HOSTNAME}/xname_${XNAME}/yname_${YNAME}"
        ODIR=$IDIR

        mkdir -p $ODIR $IDIR $RESDIR $TMPROOT

        LOG=$RESDIR/log_bash.txt

        rm -fr $LOG

        NITER=2000

        LN="TMPROOT=$TMPROOT python=$PYTHONDEFAULT IDIR=$IDIR ODIR=$ODIR RESDIR=$RESDIR it=$NITER xname=$XNAME yname=$YNAME"
        echo_log "$LN"

        CAPGEN=$RESDIR/cap_gen_data.txt

        mkdir -p $ODIR

        FDGEN=$IDIR/done_gen_data.txt

        if [[ ! -e $FDGEN ]]; then
            time_init
            echo "$FDGEN does not exist"
            CMD="$PYTHONDEFAULT Data_Generation_Main.py -o $IDIR --xname $XNAME --yname $YNAME"
            echo_log "$CMD"
            $CMD 2>&1 | tee $CAPGEN
            time_log
            echo "time=$TIME_DURATION" > $FDGEN
        fi

        cat $FDGEN >> $LOG

        N="0"
        ADIR=$RESDIR/n_${N}/it_${NITER}
        mkdir -p $ADIR
        CAPMAIN=$ADIR/cap_main.txt
        CAPEXP=$ADIR/cap_exp.txt
        FD=$ADIR/done.txt
        FDEX=$ADIR/done_ex.txt
        CMD="$PYTHONDEFAULT KnockoffGAN_Main.py -i $IDIR  -o $ADIR --it $NITER --xname $XNAME --yname $YNAME"

        if [[ ! -e $FD ]]; then
            echo_log "$CMD"
            time_init
            $CMD 2>&1 | tee $CAPMAIN
            time_log
            echo "time=$TIME_DURATION" > $FD
        fi

        if [[ ! -e $FDEX ]]; then
            mkdir -p "$ADIR/Result"
            CMD="Rscript Experiment_Main.R -i $IDIR -o $ADIR --xname $XNAME --yname $YNAME"
            echo_log "$CMD"
            time_init
            $CMD 2>&1 | tee $CAPEXP
            time_log
            echo "time=$TIME_DURATION" > $FDEX
        fi

        $PYTHONDEFAULT plot_fdr_tpr.py -i $ADIR/Result -o $ADIR/plot_fdr_tpr.jpg --xname $XNAME --yname $YNAME
    done
done
