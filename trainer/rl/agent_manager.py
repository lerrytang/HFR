"""Agent manager.

This file implements a robot manager who is responsible for creating RL agents.

"""
import common.replay_buffer
import common.rand_process
import common.network
import td3


class AgentManager(object):
    """Agent manager."""

    @classmethod
    def create_agent(cls, config):
        """Create an agent with specified type and config."""
        if config.agent == 'TD3':
            replay_buffer = common.replay_buffer.ReplayBuffer(
                config.buffer_size)
            if config.rand_process == 'gaussian':
                rand_process = common.rand_process.GaussianNoise(
                    action_dim=config.action_dim,
                    mu=config.mu,
                    sigma=config.sigma,
                    sigma_min=config.sigma_min,
                    annealing_steps=config.annealing_steps)
            else:
                rand_process = common.rand_process.OrnsteinUhlenbeck(
                    action_dim=config.action_dim,
                    theta=config.theta,
                    sigma=config.sigma,
                    init_val=config.init_val,
                    scale_min=config.scale_min,
                    annealing_steps=config.annealing_steps)

            im_encoders = {}
            if ((not config.use_actor_im_encoder) and
                    (not config.use_critic_im_encoder)):
                im_encoders['critic'] = common.network.ImageEncoder(
                    img_h=config.img_h,
                    img_w=config.img_w,
                    img_c=config.action_repeat,
                    name='c_im_encoder')
                if config.share_im_encoder:
                    im_encoders['critic_target'] = im_encoders['critic']
                else:
                    im_encoders['critic_target'] = common.network.ImageEncoder(
                        img_h=config.img_h,
                        img_w=config.img_w,
                        img_c=config.action_repeat,
                        name='c_img_encoder_target')
                im_encoders['actor'] = common.network.ImageEncoder(
                    img_h=config.img_h,
                    img_w=config.img_w,
                    img_c=config.action_repeat,
                    name='a_im_encoder')
                if config.share_im_encoder:
                    im_encoders['actor_target'] = im_encoders['actor']
                else:
                    im_encoders['actor_target'] = common.network.ImageEncoder(
                        img_h=config.img_h,
                        img_w=config.img_w,
                        img_c=config.action_repeat,
                        name='a_img_encoder_target')
            elif ((not config.use_actor_im_encoder) and
                  config.use_critic_im_encoder):
                im_encoders['critic'] = common.network.ImageEncoder(
                    img_h=config.img_h,
                    img_w=config.img_w,
                    img_c=config.action_repeat,
                    name='c_im_encoder')
                if config.share_im_encoder:
                    im_encoders['critic_target'] = im_encoders['critic']
                else:
                    im_encoders['critic_target'] = common.network.ImageEncoder(
                        img_h=config.img_h,
                        img_w=config.img_w,
                        img_c=config.action_repeat,
                        name='c_img_encoder_target')
                im_encoders['actor'] = im_encoders['critic']
                im_encoders['actor_target'] = im_encoders['critic_target']
            elif (config.use_actor_im_encoder and
                  (not config.use_critic_im_encoder)):
                im_encoders['actor'] = common.network.ImageEncoder(
                    img_h=config.img_h,
                    img_w=config.img_w,
                    img_c=config.action_repeat,
                    name='a_im_encoder')
                if config.share_im_encoder:
                    im_encoders['actor_target'] = im_encoders['actor']
                else:
                    im_encoders['actor_target'] = common.network.ImageEncoder(
                        img_h=config.img_h,
                        img_w=config.img_w,
                        img_c=config.action_repeat,
                        name='a_img_encoder_target')
                im_encoders['critic'] = im_encoders['actor']
                im_encoders['critic_target'] = im_encoders['actor_target']
            else:
                raise ValueError('Wrong config.')
            critic1 = common.network.Critic(
                action_dim=config.action_dim,
                learning_rate=config.critic_lr,
                tau=config.tau,
                img_encoder=im_encoders['critic'],
                target_img_encoder=im_encoders['critic_target'],
                train_img_encoder=(not config.use_actor_im_encoder),
                share_img_encoder=config.share_im_encoder,
                name='critic1')
            critic2 = common.network.Critic(
                action_dim=config.action_dim,
                learning_rate=config.critic_lr,
                tau=config.tau,
                img_encoder=im_encoders['critic'],
                target_img_encoder=im_encoders['critic_target'],
                train_img_encoder=False,
                share_img_encoder=True,
                name='critic2')
            actor = common.network.Actor(
                action_dim=config.action_dim,
                action_high=config.action_high,
                action_low=config.action_low,
                learning_rate=config.actor_lr,
                tau=config.tau,
                batch_size=config.batch_size,
                grad_norm_clip=config.grad_norm_clip,
                img_encoder=im_encoders['actor'],
                target_img_encoder=im_encoders['actor_target'],
                train_img_encoder=(not config.use_critic_im_encoder),
                share_img_encoder=config.share_im_encoder)
            agent = td3.TD3Agent(critic1=critic1,
                                 critic2=critic2,
                                 actor=actor,
                                 replay_buffer=replay_buffer,
                                 rand_process=rand_process,
                                 warmup_size=config.warmup_size,
                                 batch_size=config.batch_size,
                                 gamma=config.gamma,
                                 sigma=config.sigma_tilda,
                                 smoothing_regularization_cap=config.c,
                                 update_interval=config.d)
        elif config.agent == 'DDPG':
            raise NotImplementedError('DDPG not implemented')
        elif config.agent == 'C2A2':
            raise NotImplementedError('C2A2 not implemented')
        else:
            raise ValueError('Unknown robot type: {}'.format(config.agent_type))
        return agent
