import R2P.helper
import utils.make_data as md
from CMGP.models.causal_models import CMGP
from R2P.R2P_HTE import R2P_HTE


def define_estimator(type="CMGP", input_dim=None):
    ####################################################
    # Estimator
    ####################################################
    if type is "CMGP":
        """ CMGP """
        CMGP_model = CMGP(dim=input_dim, mode="CMGP")
        estimator_treat = R2P.helper.RegressorAdapter_HTE(CMGP_model, mode=1)
        estimator_control = R2P.helper.RegressorAdapter_HTE(CMGP_model, mode=0)

    return estimator_treat, estimator_control


def data_generation(data_type="SYNTH", file_path=None, noise=0.1):
    ####################################################
    # Data generation
    ####################################################
    if data_type == "SYNTH_A":
        """ SYNTH data """
        train_data = md.sample_SynthA(sample_no=300)
        x_train, w_train, y_train = train_data[0], train_data[1], train_data[2]
        treat_idx_train = w_train == 1
        control_idx_train = w_train == 0
        x1_train = train_data[0][treat_idx_train]
        x0_train = train_data[0][control_idx_train]
        y1_train = train_data[3][treat_idx_train]
        y0_train = train_data[4][control_idx_train]

        test_data = md.sample_SynthA(sample_no=1000)
        x_test, y1_test, y0_test, tau_test = test_data[0], test_data[3], test_data[4], test_data[5]
    elif data_type == "SYNTH_B":
        """ Based on the initial clinical trial of remdesivir """
        train_data, test_data = md.sample_SynthB(train_sample_no=300,
                                                 test_sample_no=1000)
        x_train, w_train, y_train = train_data[0], train_data[1], train_data[2]
        x_test, y0_test, y1_test, tau_test = test_data[0], test_data[3], test_data[4], test_data[6]

        treat_idx_train = w_train == 1
        control_idx_train = w_train == 0
        x1_train = x_train[treat_idx_train]
        x0_train = x_train[control_idx_train]
        y1_train = y_train[treat_idx_train]
        y0_train = y_train[control_idx_train]
    elif data_type == "IHDP":
        """ IHDP data """
        test_frac = 0.2
        train_data, test_data = md.sample_IHDP(file_path, test_frac=test_frac, noise=noise)
        x_train, w_train, y_train = train_data[0], train_data[1], train_data[2]
        x_test, y0_test, y1_test, tau_test = test_data[0], test_data[3], test_data[4], test_data[6]

        treat_idx_train = w_train == 1
        control_idx_train = w_train == 0
        x1_train = x_train[treat_idx_train]
        x0_train = x_train[control_idx_train]
        y1_train = y_train[treat_idx_train]
        y0_train = y_train[control_idx_train]
    elif data_type == "CPP":
        """ CPP data """
        train_data, test_data = md.sample_CPP(file_path=file_path,
                                              train_sample_no=500,
                                              test_sample_no=300)
        x_train, w_train, y_train = train_data[0], train_data[1], train_data[2]
        x_test, y0_test, y1_test, tau_test = test_data[0], test_data[3], test_data[4], test_data[6]

        treat_idx_train = w_train == 1
        control_idx_train = w_train == 0
        x1_train = x_train[treat_idx_train]
        x0_train = x_train[control_idx_train]
        y1_train = y_train[treat_idx_train]
        y0_train = y_train[control_idx_train]

    return x_train, w_train, y_train, x1_train, x0_train, y1_train, y0_train, x_test, y0_test, y1_test, tau_test


def do_R2P(estimator_treat=None, estimator_control=None, data=None, **kwargs):
    x_train, w_train, y_train, x1_train, x0_train, y1_train, y0_train, x_test, y0_test, y1_test, tau_test = data
    ####################################################
    # Robust Recursive Partitioning
    ####################################################
    if kwargs is not None:
        r2p = R2P_HTE(estimator_treat=estimator_treat, estimator_control=estimator_control, **kwargs)
    else:
        r2p = R2P_HTE(estimator_treat=estimator_treat, estimator_control=estimator_control)

    r2p.fit(x1_train, y1_train, x0_train, y0_train)
    r2p_predict = r2p.predict(x_test, test_y1=y1_test, test_y0=y0_test, test_tau=tau_test)
    r2p_predict_root = r2p.predict(x_test, test_y1=y1_test, test_y0=y0_test, test_tau=tau_test, root=True)

    return r2p, r2p_predict, r2p_predict_root
