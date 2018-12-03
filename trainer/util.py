"""Utilities."""

import numpy as np
import tensorflow as tf


def build_summaries():
    """Training and evaluation summaries."""
    # train summaries
    episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
    summary_vars = [episode_reward]
    with tf.name_scope('Training'):
        reward = tf.summary.scalar("Reward", episode_reward)
    summary_ops = tf.summary.merge([reward])
    # eval summary
    eval_episode_reward = tf.placeholder(dtype=tf.float32, shape=[])
    eval_summary_vars = [eval_episode_reward]
    with tf.name_scope('Evaluation'):
        eval_reward = tf.summary.scalar("EvalReward", eval_episode_reward)
    eval_summary_ops = tf.summary.merge([eval_reward])

    return summary_ops, summary_vars, eval_summary_ops, eval_summary_vars


def log_metrics(sess, writer, summary_ops, summary_vals, metrics, test=False):
    """Log metrics."""
    ep_cnt, ep_r, steps, actions, noises = metrics
    if test:
        tf.logging.info(
            '[TEST] Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d}'.format(ep_cnt, ep_r, ep_r / steps, steps))
    else:
        aa = np.array(actions).mean(axis=0).squeeze()
        nn = np.array(noises).mean(axis=0).squeeze()
        tf.logging.info(
            '| Episode: {:d} | Reward: {:.2f} | AvgReward: {:.2f} | '
            'Steps: {:d} | AvgAction: {} | AvgNoise: {}'.format(
                ep_cnt, ep_r, ep_r / steps, steps, aa, nn))
    summary_str = sess.run(summary_ops, feed_dict={summary_vals[0]: ep_r})
    writer.add_summary(summary_str, ep_cnt)
