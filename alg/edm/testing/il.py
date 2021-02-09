try:
    from  __head__ import *
except:
    from .__head__ import *

from agent    import BaseAgent
from agent    import OAStableAgent
from network  import StudentNetwork
from student  import (
    BaseStudent,
    EDMStudent ,
)

def save_results(
        results_file_path: str,
        repetition_num: int,
        match_mean: float,
        return_mean: float,
        return_std: float,
):
    with open(results_file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow((repetition_num, match_mean, return_mean, return_std))


def make_agent(
    env_name : str,
    alg_name : str,
) -> OAStableAgent:
    env              = gym.make(env_name)
    trajs_path       = get_trajs_path(env_name, 'expert')
    algorithm        = alg_name

    return OAStableAgent(
        env          = env       ,
        trajs_path   = trajs_path,
        algorithm    = algorithm ,
    )


def make_student(
        run_seed: int, config
) -> BaseStudent:

    env = gym.make(config['ENV'])
    trajs_path = get_trajs_path(config['ENV'], 'student_' + config['ALG'], run_seed)
    model_path = get_model_path(config['ENV'], 'student_' + config['ALG'], run_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    run_seed = run_seed
    batch_size = config['BATCH_SIZE']
    buffer_size_in_trajs = config['NUM_TRAJS_GIVEN']
    teacher = make_agent(config['ENV'], config['EXPERT_ALG']);
    teacher.load_pretrained()
    qvalue_function = StudentNetwork(
        in_dim=state_dim,
        out_dim=action_dim,
        width=config['MLP_WIDTHS'],
    )

    adam_alpha = config['ADAM_ALPHA']
    adam_betas = config['ADAM_BETAS']
    sgld_buffer_size = config['SGLD_BUFFER_SIZE']
    sgld_learn_rate = config['SGLD_LEARN_RATE']
    sgld_noise_coef = config['SGLD_NOISE_COEF']
    sgld_num_steps = config['SGLD_NUM_STEPS']
    sgld_reinit_freq = config['SGLD_REINIT_FREQ']

    return EDMStudent(
        env=env,
        trajs_path=trajs_path,
        model_path=model_path,
        run_seed=run_seed,
        batch_size=batch_size,
        buffer_size_in_trajs=buffer_size_in_trajs,
        teacher=teacher,
        qvalue_function=qvalue_function,
        adam_alpha=adam_alpha,
        adam_betas=adam_betas,
        sgld_buffer_size=sgld_buffer_size,
        sgld_learn_rate=sgld_learn_rate,
        sgld_noise_coef=sgld_noise_coef,
        sgld_num_steps=sgld_num_steps,
        sgld_reinit_freq=sgld_reinit_freq,
    )

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v1')
    parser.add_argument("--num_trajectories", default=10, type=int)
    parser.add_argument("--trial", default=0, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = init_arg()

    config = {
        'ENV': args.env_name,
        'ALG': 'EDMStudent',
        'NUM_TRAJS_GIVEN': args.num_trajectories,
        'NUM_STEPS_TRAIN': 10000,
        'NUM_TRAJS_VALID': 300,
        'NUM_REPETITIONS': 10,

        'BATCH_SIZE': 32,
        'MLP_WIDTHS': 64,
        'ADAM_ALPHA': 1e-3,
        'ADAM_BETAS': [0.9, 0.999],

        'SGLD_BUFFER_SIZE': 10000,
        'SGLD_LEARN_RATE': 0.01,
        'SGLD_NOISE_COEF': 0.01,
        'SGLD_NUM_STEPS': 20,
        'SGLD_REINIT_FREQ': 0.05,
    }

    config['EXPERT_ALG'] = yaml.load(open('testing/config.yml'), \
        Loader = yaml.FullLoader)[config['ENV']]

    print("Config: %s" % config)

    TRIAL = args.trial
    print ("Trial number %s" % TRIAL)


    results_dir_base = 'testing/results/'
    results_dir = os.path.join(results_dir_base, config['ENV'], str(config['NUM_TRAJS_GIVEN']), config['ALG'])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config_file = 'trial_' + str(TRIAL) + '_' + 'config.pkl'

    results_file_name = 'trial_' + str(TRIAL) + '_' + 'results.csv'
    results_file_path = os.path.join(results_dir, results_file_name)

    if os.path.exists(os.path.join(results_dir, config_file)):
       raise NameError('CONFIG file already exists %s. Choose a different trial number.' % config_file)
    pickle.dump(config, open(os.path.join(results_dir, config_file), 'wb'))

    for run_seed in range(config['NUM_REPETITIONS']):
        print ("Run %s out of %s" % (run_seed + 1, config['NUM_REPETITIONS']))
        student = make_student(run_seed, config)
        student.train(num_updates=config['NUM_STEPS_TRAIN'])
        action_match, return_mean, return_std = student.test(num_episodes=config['NUM_TRAJS_VALID'])
        result = (action_match, return_mean, return_std)
        print("Reward for run %s: %s" % (run_seed, return_mean))
        save_results(results_file_path, run_seed, action_match, return_mean, return_std)


    results_trial = pd.read_csv(
        'testing/results/' + config['ENV'] + '/' + str(config['NUM_TRAJS_GIVEN']) + '/EDMStudent/trial_' +
        str(TRIAL) + '_results.csv', header=None)

    print("Average reward for 10 repetitions: %s" % np.mean(results_trial[2].values))




