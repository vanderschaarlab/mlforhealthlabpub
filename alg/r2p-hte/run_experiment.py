import warnings

warnings.filterwarnings('ignore')

import argparse

import utils.experiments_ilb as exp
import utils.utils as util


def get_parser():
    parser = argparse.ArgumentParser(description="Experiments for 'Robust Recursive Partitioning for"
                                                 "Heterogeneous Treatmnet Effects with Uncertainty Quantification'")

    parser.add_argument('--data', required=True, help='types of dataset {SYNTH_A, SYNTH_B, IHDP, CPP}')
    parser.add_argument('--file_path', required=False, default=None, help='file path of dataset')
    parser.add_argument('--max_depth', required=False, type=int, default=-1,
                        help='maximum depth of partition (-1 for no limits)')
    parser.add_argument('--min_size', required=False, type=int, default=10,
                        help='minimum number of samples for each subgroup')
    parser.add_argument('--miscoverage', required=False, type=float, default=0.05, help='target miscoverage rate')
    parser.add_argument('--weight', required=False, type=float, default=0.5, help='weight parameter (lambda)')
    parser.add_argument('--gamma', required=False, type=float, default=0.05, help='weight parameter (lambda)')

    return parser


def main(args=None):
    iter = 50
    parser = get_parser()
    args = parser.parse_args(args)

    data_list = ['SYNTH_A', 'SYNTH_B', 'IHDP', 'CPP']

    if args.data not in data_list:
        raise ValueError('Invalid data type.')
    if args.data in ['IHDP', 'CPP'] and args.file_path == None:
        raise ValueError('For IHDP and CPP datasets, file path is required.')

    output = {}
    R2P = util.init_output_dict(output, name="R2P")
    R2P_Root = util.init_output_dict(output, name="R2P-Root")

    for i in range(iter):
        FLAG = True
        while FLAG:
            try:
                dataset = exp.data_generation(data_type=args.data,
                                              file_path=args.file_path)
                CMGP_treat, CMGP_control = exp.define_estimator(type="CMGP", input_dim=dataset[0].shape[1])
                r2p, r2p_predict, r2p_predict_root = \
                    exp.do_R2P(estimator_treat=CMGP_treat, estimator_control=CMGP_control, data=dataset,
                               min_size=args.min_size,
                               max_depth=args.max_depth,
                               significance=args.miscoverage,
                               weight=args.weight,
                               gamma=args.gamma)
                util.update_output_dict(output, R2P, r2p_predict, name="R2P")
                util.update_output_dict(output, R2P_Root, r2p_predict_root, name="R2P-Root")

                FLAG = False
            except Exception as ex:
                print('Estimator internal error:', ex)
                pass

    print(
        '== Results ============================================================================')
    print(
        f'|            |     V^ACROSS    |       V^IN      |  NUM SUBGROUPS  |     CI WIDTH    |')
    util.print_summary(R2P, name="R2P")
    util.print_gain(R2P, R2P_Root)


if __name__ == '__main__':
    main()
