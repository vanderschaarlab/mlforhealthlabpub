from sklearn.model_selection import train_test_split

from R2P.helper import IcpRegressor_r2p
from R2P.helper import RegressorNc_r2p
from R2P.r2p_utils import *
from nonconformist.nc import AbsErrorErrFunc


class R2P_HTE:
    def __init__(self, estimator_treat=None, estimator_control=None,
                 max_depth=-1, min_size=10,
                 conformal_mode="SCR", params_qf=None,
                 significance=0.05, weight=0.5, gamma=0.05,
                 sig_for_split=0.8,
                 seed=None):

        self.root = None
        self.max = -np.inf
        self.min = np.inf
        self.num_leaves = 0
        self.curr_leaves = 0
        self.estimator_treat = estimator_treat
        self.estimator_control = estimator_control

        self.max_depth = max_depth
        self.min_size = min_size

        self.conformal_mode = conformal_mode
        self.params_qf = params_qf
        self.significance = 1 - np.sqrt(1 - significance)
        self.sig_for_split = 1 - np.sqrt(1 - sig_for_split)
        self.weight = weight
        self.gamma = gamma
        self.seed = seed
        self.eval_func = self.conf_homo

        self.tree_depth = 0
        self.obj = 0.0
        self.start = 0.0
        self.time = 0.0

    class Node:
        def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, leaf=False, leaf_num=None,
                     obj=0.0, homogeneity=0.0, intv_len=0.0,
                     est_treat_treat=None, est_treat_control=None,
                     est_control_treat=None, est_control_control=None,
                     conf_pred_treat=None, confl_pred_control=None,
                     cal_scores_treat=None, cal_scores_control=None,
                     node_depth=0):
            self.col = col  # the column of the feature used for splitting
            self.value = value  # the value that splits the data

            self.est_treat_treat = est_treat_treat
            self.est_treat_control = est_treat_control
            self.est_control_treat = est_control_treat
            self.est_control_control = est_control_control

            self.conf_pred_treat = conf_pred_treat
            self.conf_pred_control = confl_pred_control

            self.cal_scores_treat = cal_scores_treat
            self.cal_scores_control = cal_scores_control

            self.obj = obj
            self.intv_len = intv_len
            self.homogeneity = homogeneity

            self.true_branch = true_branch  # pointer to node for true branch
            self.false_branch = false_branch  # pointer to node for false branch
            self.leaf = leaf  # true/false if leaf or not
            self.leaf_num = leaf_num  # the leaf label

            self.node_depth = node_depth

    def fit(self, rows_treat, labels_treat, rows_control, labels_control):
        if rows_treat.shape[0] == 0:
            return self.Node()

        if self.seed is not None:
            np.random.seed(self.seed)

        # split for conformal regression
        train_rows_treat, val_rows_treat, train_outcome_treat, val_labels_treat = \
            train_test_split(rows_treat, labels_treat, shuffle=True, test_size=0.5)
        train_rows_control, val_rows_control, train_outcome_control, val_labels_control = \
            train_test_split(rows_control, labels_control, shuffle=True, test_size=0.5)

        # check estimator internal error
        error_no_tmp = 0
        FIT_FLAG = True
        while (FIT_FLAG):
            x_train = np.concatenate([train_rows_treat, train_rows_control])
            y_train = np.concatenate([train_outcome_treat, train_outcome_control])
            w_train = np.zeros(x_train.shape[0])
            w_train[0:train_rows_treat.shape[0]] = 1
            FIT_FLAG = self.estimator_treat.model.fit(x_train, y_train, w_train)
            error_no_tmp = error_no_tmp + 1
            if error_no_tmp > 2:
                # error occur request new datasets
                raise Exception('Too many errors occur in internal estimator.')

        # do conformal prediction
        total_val_no_treat = val_rows_treat.shape[0]
        total_val_no_control = val_rows_control.shape[0]

        if self.conformal_mode == "SCR":
            nc_treat = RegressorNc_r2p(self.estimator_treat, AbsErrorErrFunc())
            nc_control = RegressorNc_r2p(self.estimator_control, AbsErrorErrFunc())

        icp_treat = IcpRegressor_r2p(nc_treat)
        icp_treat.fit(train_rows_treat, train_outcome_treat.reshape((train_outcome_treat.shape[0], 1)))
        icp_treat.calibrate(val_rows_treat, val_labels_treat)
        cal_scores_treat = icp_treat.cal_scores

        icp_control = IcpRegressor_r2p(nc_control)
        icp_control.fit(train_rows_control, train_outcome_control.reshape((train_outcome_control.shape[0], 1)))
        icp_control.calibrate(val_rows_control, val_labels_control)
        cal_scores_control = icp_control.cal_scores

        val_est_treat_treat = self.estimator_treat.predict(val_rows_treat)
        val_est_treat_control = self.estimator_control.predict(val_rows_treat)
        val_est_treat_CATE = val_est_treat_treat - val_est_treat_control
        val_est_control_treat = self.estimator_treat.predict(val_rows_control)
        val_est_control_control = self.estimator_control.predict(val_rows_control)
        val_est_control_CATE = val_est_control_treat - val_est_control_control
        val_est = np.concatenate([val_est_treat_CATE, val_est_control_CATE])
        est_mean = float(np.mean(val_est))

        # calculate partition measure
        val_rows = np.concatenate([val_rows_treat, val_rows_control])
        val_rows_est_treat = np.concatenate([val_est_treat_treat, val_est_control_treat])
        val_rows_est_control = np.concatenate([val_est_treat_control, val_est_control_control])

        intv_treat = icp_treat.predict(val_rows, significance=self.significance, est_input=val_rows_est_treat)
        intv_control = icp_control.predict(val_rows, significance=self.significance, est_input=val_rows_est_control)
        intv = self.get_TE_CI(intv_treat, intv_control)
        intv_len = np.mean(intv[:, 1] - intv[:, 0])

        intv_treat_split = icp_treat.predict(val_rows, significance=self.sig_for_split, est_input=val_rows_est_treat)
        intv_control_split = icp_control.predict(val_rows, significance=self.sig_for_split,
                                                 est_input=val_rows_est_control)
        intv_split = self.get_TE_CI(intv_treat_split, intv_control_split)
        intv_len_split = np.mean(intv_split[:, 1] - intv_split[:, 0])

        obj, intv_measure, homogeneity, obj_real = \
            self.eval_func(intv, intv_split, est_mean, total_val_no_treat, total_val_no_control)

        if self.seed is not None:
            np.random.seed(self.seed)

        self.obj = obj
        self.curr_leaves = 1
        self.root = self.Node(col=-1, value=None, obj=obj, homogeneity=homogeneity, intv_len=intv_len,
                              est_treat_treat=val_est_treat_treat, est_treat_control=val_est_treat_control,
                              est_control_treat=val_est_control_treat, est_control_control=val_est_control_control,
                              conf_pred_treat=icp_treat, confl_pred_control=icp_control,
                              cal_scores_treat=cal_scores_treat, cal_scores_control=cal_scores_control, node_depth=0)

        self.root = self.fit_r(rows_treat, labels_treat, rows_control, labels_control, curr_depth=0, node=self.root,
                               val_rows_treat=val_rows_treat, val_labels_treat=val_labels_treat,
                               val_rows_control=val_rows_control, val_labels_control=val_labels_control,
                               total_val_no_treat=total_val_no_treat, total_val_no_control=total_val_no_control)

    def fit_r(self, rows_treat, labels_treat, rows_control, labels_control, curr_depth=0, node=None,
              val_rows_treat=None, val_labels_treat=None, val_rows_control=None, val_labels_control=None,
              total_val_no_treat=None, total_val_no_control=None):
        if rows_treat.shape[0] == 0:
            return node

        if curr_depth > self.tree_depth:
            self.tree_depth = curr_depth

        if self.max_depth == curr_depth:
            # node leaf number
            self.num_leaves += 1
            # add node leaf number to node class
            node.leaf_num = self.num_leaves
            node.leaf = True
            return node

        best_gain = 0.0
        best_attribute = None

        best_tb_obj = 0.0
        best_fb_obj = 0.0

        best_tb_intv_len = 0.0
        best_fb_intv_len = 0.0

        best_tb_homo = 0.0
        best_fb_homo = 0.0

        curr_depth += 1

        column_count = rows_treat.shape[1]
        rows = np.concatenate([rows_treat, rows_control])
        for col in range(0, column_count):
            # unique values
            unique_vals = np.unique(rows[:, col])

            for value in unique_vals:
                # binary treatment splitting
                (tb_val_set_treat, fb_val_set_treat,
                 tb_val_y_treat, fb_val_y_treat,
                 tb_val_idx_treat, fb_val_idx_treat) = \
                    divide_set(val_rows_treat, val_labels_treat, col, value)
                (tb_val_set_control, fb_val_set_control,
                 tb_val_y_control, fb_val_y_control,
                 tb_val_idx_control, fb_val_idx_control) = \
                    divide_set(val_rows_control, val_labels_control, col, value)

                if tb_val_set_treat.shape[0] < self.min_size or tb_val_set_control.shape[0] < self.min_size or \
                        fb_val_set_treat.shape[0] < self.min_size or fb_val_set_control.shape[0] < self.min_size:
                    continue
                if tb_val_set_treat.shape[0] == 0 or tb_val_set_control.shape[0] == 0 \
                        or fb_val_set_treat.shape[0] == 0 or fb_val_set_control.shape[0] == 0:
                    continue

                tb_cal_scores_treat = {0: node.cal_scores_treat[0][tb_val_idx_treat]}
                tb_cal_scores_control = {0: node.cal_scores_control[0][tb_val_idx_control]}

                tb_val_set = np.concatenate([tb_val_set_treat, tb_val_set_control])
                tb_val_set_est_treat_treat = node.est_treat_treat[tb_val_idx_treat]
                tb_val_set_est_control_treat = node.est_control_treat[tb_val_idx_control]
                tb_val_set_est_treat = np.concatenate([tb_val_set_est_treat_treat, tb_val_set_est_control_treat])
                tb_val_set_est_treat_control = node.est_treat_control[tb_val_idx_treat]
                tb_val_set_est_control_control = node.est_control_control[tb_val_idx_control]
                tb_val_set_est_control = np.concatenate([tb_val_set_est_treat_control, tb_val_set_est_control_control])
                tb_intv_treat = node.conf_pred_treat.predict_given_scores(tb_val_set,
                                                                          significance=self.significance,
                                                                          cal_scores=tb_cal_scores_treat,
                                                                          est_input=tb_val_set_est_treat)
                tb_intv_control = node.conf_pred_control.predict_given_scores(tb_val_set,
                                                                              significance=self.significance,
                                                                              cal_scores=tb_cal_scores_control,
                                                                              est_input=tb_val_set_est_control)
                tb_intv = self.get_TE_CI(tb_intv_treat, tb_intv_control)
                tb_intv_len = np.mean(tb_intv[:, 1] - tb_intv[:, 0])

                tb_intv_treat_split = node.conf_pred_treat.predict_given_scores(tb_val_set,
                                                                                significance=self.sig_for_split,
                                                                                cal_scores=tb_cal_scores_treat,
                                                                                est_input=tb_val_set_est_treat)
                tb_intv_control_split = node.conf_pred_control.predict_given_scores(tb_val_set,
                                                                                    significance=self.sig_for_split,
                                                                                    cal_scores=tb_cal_scores_control,
                                                                                    est_input=tb_val_set_est_control)
                tb_intv_split = self.get_TE_CI(tb_intv_treat_split, tb_intv_control_split)
                tb_intv_len_split = np.mean(tb_intv_split[:, 1] - tb_intv_split[:, 0])

                tb_val_est = tb_val_set_est_treat - tb_val_set_est_control
                tb_est_mean = float(np.mean(tb_val_est))

                fb_cal_scores_treat = {0: node.cal_scores_treat[0][fb_val_idx_treat]}
                fb_cal_scores_control = {0: node.cal_scores_control[0][fb_val_idx_control]}

                fb_val_set = np.concatenate([fb_val_set_treat, fb_val_set_control])
                fb_val_set_est_treat_treat = node.est_treat_treat[fb_val_idx_treat]
                fb_val_set_est_control_treat = node.est_control_treat[fb_val_idx_control]
                fb_val_set_est_treat = np.concatenate([fb_val_set_est_treat_treat, fb_val_set_est_control_treat])
                fb_val_set_est_treat_control = node.est_treat_control[fb_val_idx_treat]
                fb_val_set_est_control_control = node.est_control_control[fb_val_idx_control]
                fb_val_set_est_control = np.concatenate([fb_val_set_est_treat_control, fb_val_set_est_control_control])

                fb_intv_treat = node.conf_pred_treat.predict_given_scores(fb_val_set,
                                                                          significance=self.significance,
                                                                          cal_scores=fb_cal_scores_treat,
                                                                          est_input=fb_val_set_est_treat)
                fb_intv_control = node.conf_pred_control.predict_given_scores(fb_val_set,
                                                                              significance=self.significance,
                                                                              cal_scores=fb_cal_scores_control,
                                                                              est_input=fb_val_set_est_control)
                fb_intv = self.get_TE_CI(fb_intv_treat, fb_intv_control)
                fb_intv_len = np.mean(fb_intv[:, 1] - fb_intv[:, 0])

                fb_intv_treat_split = node.conf_pred_treat.predict_given_scores(fb_val_set,
                                                                                significance=self.sig_for_split,
                                                                                cal_scores=fb_cal_scores_treat,
                                                                                est_input=fb_val_set_est_treat)
                fb_intv_control_split = node.conf_pred_control.predict_given_scores(fb_val_set,
                                                                                    significance=self.sig_for_split,
                                                                                    cal_scores=fb_cal_scores_control,
                                                                                    est_input=fb_val_set_est_control)
                fb_intv_split = self.get_TE_CI(fb_intv_treat_split, fb_intv_control_split)
                fb_intv_len_split = np.mean(fb_intv_split[:, 1] - fb_intv_split[:, 0])

                fb_val_est = fb_val_set_est_treat - fb_val_set_est_control
                fb_est_mean = float(np.mean(fb_val_est))

                tb_obj, tb_intv_measure, tb_homogeneity, tb_obj_real = \
                    self.eval_func(tb_intv, tb_intv_split, tb_est_mean, total_val_no_treat, total_val_no_control)
                fb_obj, fb_intv_measure, fb_homogeneity, fb_obj_real = \
                    self.eval_func(fb_intv, fb_intv_split, fb_est_mean, total_val_no_treat, total_val_no_control)

                # criterion
                gain = node.obj - tb_obj - fb_obj

                if gain > best_gain:
                    best_gain = gain
                    best_attribute = [col, value]
                    best_tb_obj, best_fb_obj = tb_obj, fb_obj
                    best_tb_obj_real, best_fb_obj_real = tb_obj_real, fb_obj_real
                    best_tb_intv_len, best_fb_intv_len = tb_intv_len, fb_intv_len
                    best_tb_homo, best_fb_homo = tb_homogeneity, fb_homogeneity
                    best_tb_set_treat, best_fb_set_treat = tb_val_set_treat, fb_val_set_treat
                    best_tb_y_treat, best_fb_y_treat = tb_val_y_treat, fb_val_y_treat
                    best_tb_set_control, best_fb_set_control = tb_val_set_control, fb_val_set_control
                    best_tb_y_control, best_fb_y_control = tb_val_y_control, fb_val_y_control
                    best_tb_val_est_treat_treat, best_fb_val_est_treat_treat = tb_val_set_est_treat_treat, fb_val_set_est_treat_treat
                    best_tb_val_est_treat_control, best_fb_val_est_treat_control = tb_val_set_est_treat_control, fb_val_set_est_treat_control
                    best_tb_val_est_control_treat, best_fb_val_est_control_treat = tb_val_set_est_control_treat, fb_val_set_est_control_treat
                    best_tb_val_est_control_control, best_fb_val_est_control_control = tb_val_set_est_control_control, fb_val_set_est_control_control
                    best_tb_cal_scores_treat, best_fb_cal_scores_treat = tb_cal_scores_treat, fb_cal_scores_treat
                    best_tb_cal_scores_control, best_fb_cal_scores_control = tb_cal_scores_control, fb_cal_scores_control

        if best_gain > self.gamma * node.obj:
            node.col = best_attribute[0]
            node.value = best_attribute[1]

            self.curr_leaves = self.curr_leaves + 1

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            tb = self.Node(obj=best_tb_obj_real, homogeneity=best_tb_homo, intv_len=best_tb_intv_len,
                           est_treat_treat=best_tb_val_est_treat_treat, est_treat_control=best_tb_val_est_treat_control,
                           est_control_treat=best_tb_val_est_control_treat,
                           est_control_control=best_tb_val_est_control_control,
                           conf_pred_treat=node.conf_pred_treat,
                           confl_pred_control=node.conf_pred_control,
                           cal_scores_treat=best_tb_cal_scores_treat, cal_scores_control=best_tb_cal_scores_control,
                           node_depth=curr_depth)
            fb = self.Node(obj=best_fb_obj_real, homogeneity=best_fb_homo, intv_len=best_fb_intv_len,
                           est_treat_treat=best_fb_val_est_treat_treat, est_treat_control=best_fb_val_est_treat_control,
                           est_control_treat=best_fb_val_est_control_treat,
                           est_control_control=best_fb_val_est_control_control,
                           conf_pred_treat=node.conf_pred_treat,
                           confl_pred_control=node.conf_pred_control,
                           cal_scores_treat=best_fb_cal_scores_treat, cal_scores_control=best_fb_cal_scores_control,
                           node_depth=curr_depth)

            node.true_branch = self.fit_r(rows_treat, labels_treat, rows_control, labels_control,
                                          curr_depth=curr_depth, node=tb,
                                          val_rows_treat=best_tb_set_treat, val_labels_treat=best_tb_y_treat,
                                          val_rows_control=best_tb_set_control, val_labels_control=best_tb_y_control,
                                          total_val_no_treat=total_val_no_treat,
                                          total_val_no_control=total_val_no_control)
            node.false_branch = self.fit_r(rows_treat, labels_treat, rows_control, labels_control,
                                           curr_depth=curr_depth, node=fb,
                                           val_rows_treat=best_fb_set_treat, val_labels_treat=best_fb_y_treat,
                                           val_rows_control=best_fb_set_control, val_labels_control=best_fb_y_control,
                                           total_val_no_treat=total_val_no_treat,
                                           total_val_no_control=total_val_no_control)

            return node
        else:
            # node leaf number
            self.num_leaves += 1
            # add node leaf number to node class
            node.leaf_num = self.num_leaves
            node.leaf = True
            return node

    def conf_homo(self, intv, intv_homo, est_mean, total_val_no_treat, total_val_no_control):
        num_samples = intv.shape[0]
        y_lower = intv_homo[:, 0] - est_mean
        y_upper = est_mean - intv_homo[:, 1]
        homogeneity = (np.sum(y_lower.clip(min=0)) + np.sum(y_upper.clip(min=0))) / \
                      (total_val_no_treat + total_val_no_control)
        intv_measure = np.sum(intv[:, 1] - intv[:, 0]) / (total_val_no_treat + total_val_no_control)

        obj = self.weight * intv_measure + (1 - self.weight) * homogeneity
        obj_real = self.weight * np.mean(intv[:, 1] - intv[:, 0]) + (1 - self.weight) * homogeneity

        return obj, intv_measure, homogeneity, obj_real

    def get_TE_CI(self, intv_treat, intv_control):
        intv = np.zeros(np.shape(intv_treat))
        if len(np.shape(intv_treat)) > 1:
            intv[:, 0] = intv_treat[:, 0] - intv_control[:, 1]
            intv[:, 1] = intv_treat[:, 1] - intv_control[:, 0]
        else:
            intv[0] = intv_treat[0] - intv_control[1]
            intv[1] = intv_treat[1] - intv_control[0]

        return intv

    def predict(self, test_data, test_y1=None, test_y0=None, test_tau=None, root=False):

        def classify_r(node, observation):
            if node.leaf:
                conf_intv_treat = node.conf_pred_treat.predict_given_scores(observation,
                                                                            significance=self.significance,
                                                                            cal_scores=node.cal_scores_treat)
                conf_intv_control = node.conf_pred_control.predict_given_scores(observation,
                                                                                significance=self.significance,
                                                                                cal_scores=node.cal_scores_control)
                conf_intv = self.get_TE_CI(conf_intv_treat, conf_intv_control)
                return node.leaf_num, conf_intv_treat, conf_intv_control, conf_intv
            else:
                v = observation[:, node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            return classify_r(branch, observation)

        if len(test_data.shape) == 1:
            leaf_results = classify_r(self.root, test_data)
            return leaf_results

        num_test = test_data.shape[0]

        leaf_results = np.zeros(num_test)
        predict = np.zeros(num_test)
        predict_treat = np.zeros(num_test)
        predict_control = np.zeros(num_test)
        conf_intv_treat = np.zeros([num_test, 2])
        conf_intv_control = np.zeros([num_test, 2])
        conf_intv = np.zeros([num_test, 2])

        if not root:
            CATE_intv_cov = 0
            pehe = 0
            for i in range(num_test):
                test_example = test_data[i, :]
                test_example = test_example.reshape(1, -1)
                leaf_results[i], conf_intv_treat[i, :], conf_intv_control[i, :], conf_intv[i, :] = \
                    classify_r(self.root, test_example)
                predict_treat[i] = self.estimator_treat.predict(test_example)
                predict_control[i] = self.estimator_control.predict(test_example)
                predict[i] = predict_treat[i] - predict_control[i]
                pehe = pehe + (predict[i] - test_tau[i]) ** 2

                if conf_intv[i, 1] >= test_tau[i] > conf_intv[i, 0]:
                    CATE_intv_cov = CATE_intv_cov + 1
            num_leaves = self.num_leaves
            within_var = get_within_var(self.num_leaves, leaf_results, test_tau)
            across_var = get_across_var(self.num_leaves, leaf_results, test_tau)
        else:
            CATE_intv_cov = 0
            pehe = 0
            for i in range(num_test):
                test_example = test_data[i, :]
                test_example = test_example.reshape(1, -1)
                leaf_results[i] = 0
                conf_intv_treat[i, :] = \
                    self.root.conf_pred_treat.predict_given_scores(test_example,
                                                                   significance=self.significance,
                                                                   cal_scores=self.root.cal_scores_treat)
                conf_intv_control[i, :] = \
                    self.root.conf_pred_control.predict_given_scores(test_example,
                                                                     significance=self.significance,
                                                                     cal_scores=self.root.cal_scores_control)
                conf_intv[i, :] = self.get_TE_CI(conf_intv_treat[i, :], conf_intv_control[i, :])
                predict_treat[i] = self.estimator_treat.predict(test_example)
                predict_control[i] = self.estimator_control.predict(test_example)
                predict[i] = predict_treat[i] - predict_control[i]
                pehe = pehe + (predict[i] - test_tau[i]) ** 2

                if conf_intv[i, 1] >= test_tau[i] > conf_intv[i, 0]:
                    CATE_intv_cov = CATE_intv_cov + 1
            num_leaves = 1
            within_var = np.var(test_tau)
            across_var = 0

        CATE_intv_cov = CATE_intv_cov / num_test
        pehe = pehe / num_test
        pehe = np.sqrt(pehe)
        if pehe > 0.5:
            raise Exception('Too many errors occur in internal estimator.')

        return predict, leaf_results, conf_intv, CATE_intv_cov, pehe, num_leaves, within_var, across_var
