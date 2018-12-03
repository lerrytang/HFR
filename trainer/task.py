"""Train an actor-critic agent in the human following environment. """

import argparse
from env.hfr_env_v01 import HumanFollowingGymEnv
from env.hfr_env_v01 import IMAGE_H, IMAGE_W
from gym import wrappers
import robot.robot_manager
import rl.agent_manager
import os
from tensorflow.contrib.training.python.training import hparam
import tensorflow as tf
import util


def test(sess, env, agent, config):
    """Test the trained agent"""
    tf.logging.info('Testing ...')
    s = env.reset()
    ep_reward = 0
    ep_steps = 0
    done = False

    while not done:
        if ep_steps < config.rand_steps:
            action = env.action_space.sample()
        else:
            action = agent.action(sess, s)
        s2, r, done, info = env.step(action.squeeze().tolist())
        ep_reward += r
        ep_steps += 1
        s = s2
    return ep_reward, ep_steps


def train(config):

    log_dir = os.path.join(config.job_dir, 'log')
    video_dir = os.path.join(config.job_dir, 'video')
    model_path = os.path.join(
        config.job_dir, 'model/{}.ckpt'.format(config.agent))

    env = HumanFollowingGymEnv(robot_type=robot.robot_manager.RobotType.R2D2)
    config.action_dim = env.action_dim
    config.action_high = env.action_space.high
    config.action_low = env.action_space.low
    config.state_dim = env.state_dim
    config.img_h = IMAGE_H
    config.img_w = IMAGE_W
    if config.record_video:
        eval_interval = config.eval_interval
        env = wrappers.Monitor(
            env, video_dir,
            video_callable=lambda ep: (ep + 1 - (ep + 1) / eval_interval
                                       ) % eval_interval == 0)

    agent = rl.agent_manager.AgentManager.create_agent(config)

    (summary_ops, summary_vars,
     eval_summary_ops, eval_summary_vars) = util.build_summaries()

    with tf.Session() as sess:

        saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        agent.initialize(sess)
        global_step = 0
        tf.logging.info('Start to train {} ...'.format(config.agent))
        for i in xrange(config.max_episodes):

            s = env.reset()
            ep_reward = 0
            ep_steps = 0
            noises = []
            actions = []
            done = False

            while not done:
                if ep_steps < config.rand_steps:
                    action = env.action_space.sample()
                else:
                    action, action_org, noise = agent.action_with_noise(sess, s)
                    noises.append(noise)
                    actions.append(action_org)
                action = action.squeeze()

                s2, r, done, info = env.step(action.tolist())
                ep_reward += r
                ep_steps += 1
                global_step += 1
                agent.store_experience(s, action, r, done, s2)

                agent.train(sess, global_step)
                s = s2

                if done:
                    ep_cnt = i + 1
                    util.log_metrics(sess,
                                     writer,
                                     summary_ops,
                                     summary_vars,
                                     metrics=(ep_cnt,
                                              ep_reward,
                                              ep_steps,
                                              actions,
                                              noises))
                    if ep_cnt % config.eval_interval == 0:
                        eval_ep_reward, eval_ep_steps = test(
                            sess, env, agent, config)
                        eval_ep_cnt = ep_cnt / config.eval_interval
                        util.log_metrics(sess,
                                         writer,
                                         eval_summary_ops,
                                         eval_summary_vars,
                                         metrics=(eval_ep_cnt,
                                                  eval_ep_reward,
                                                  eval_ep_steps,
                                                  None,
                                                  None),
                                         test=True)
                        ckpt_path = saver.save(sess,
                                               model_path,
                                               global_step=global_step)
                        tf.logging.info('Model saved to {}'.format(ckpt_path))
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--critic-lr',
        help='critic learning rate',
        type=float,
        default=2e-4)
    parser.add_argument(
        '--actor-lr',
        help='actor learning rate',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--gamma',
        help='reward discounting factor',
        type=float,
        default=0.99)
    parser.add_argument(
        '--tau',
        help='target network update ratio',
        type=float,
        default=0.001)
    parser.add_argument(
        '--sigma',
        help='exploration noise standard deviation',
        type=float,
        default=0.02)
    parser.add_argument(
        '--sigma-min',
        help='minimum exploration noise standard deviation',
        type=float,
        default=0.01)
    parser.add_argument(
        '--sigma-tilda',
        help='noise standard deviation for smoothing regularization',
        type=float,
        default=0.05)
    parser.add_argument(
        '--c',
        help='noise cap',
        type=float,
        default=0.15)
    parser.add_argument(
        '--grad-norm-clip',
        help='maximum allowed gradient norm',
        type=float,
        default=5.0)
    parser.add_argument(
        '--rand-process',
        help='type of random process, supported types are [gaussian|ou]',
        default='ou')
    parser.add_argument(
        '--annealing-steps',
        help='steps to anneal noise',
        type=int,
        default=0)
    parser.add_argument(
        '--mu',
        help='mu for Ornstein Uhlenbeck process',
        type=float,
        default=0.0)
    parser.add_argument(
        '--theta',
        help='theta for Ornstein Uhlenbeck process',
        type=float,
        default=0.005)
    parser.add_argument(
        '--init-val',
        help='init_val for Ornstein Uhlenbeck process',
        type=float,
        default=0.0)
    parser.add_argument(
        '--scale-min',
        help='scale_min for Ornstein Uhlenbeck process',
        type=float,
        default=0.01)
    parser.add_argument(
        '--buffer-size',
        help='replay buffer size',
        type=int,
        default=1000000)
    parser.add_argument(
        '--d',
        help='target update interval',
        type=int,
        default=2)
    parser.add_argument(
        '--warmup-size',
        help='warm up buffer size',
        type=int,
        default=100)
    parser.add_argument(
        '--batch-size',
        help='mini-batch size',
        type=int,
        default=32)
    parser.add_argument(
        '--rand-steps',
        help='number of steps to user random actions in a new episode',
        type=int,
        default=10)
    parser.add_argument(
        '--max-episodes',
        help='maximum number of episodes to train',
        type=int,
        default=3000)
    parser.add_argument(
        '--eval-interval',
        help='interval to test',
        type=int,
        default=2)
    parser.add_argument(
        '--max-to-keep',
        help='number of model generations to keep',
        type=int,
        default=5)
    parser.add_argument(
        '--agent',
        help='type of agent, one of [DDPG|TD3|C2A2]',
        default='TD3')
    parser.add_argument(
        '--use-critic-im-encoder',
        help='use critic\'s image encoder',
        action='store_true')
    parser.add_argument(
        '--use-actor-im-encoder',
        help='use actor\'s image encoder',
        action='store_true')
    parser.add_argument(
        '--share-im-encoder',
        help='share image encoder with target network',
        action='store_true')
    parser.add_argument(
        '--job-dir',
        help='dir to save logs and videos',
        default='./results')
    parser.add_argument(
        '--record-video',
        help='whether to record video when testing',
        action='store_true')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    args, _ = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    for k, v in args.__dict__.iteritems():
        tf.logging.info('{}: {}'.format(k, v))

    train(hparam.HParams(**args.__dict__))
