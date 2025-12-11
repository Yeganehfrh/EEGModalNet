import torch
from keras import layers
import keras
from .common_v0 import convBlock, ChannelMerger, SelfAttention1D, LearnablePositionalEmbedding, SubjectLayers_FiLM, FiLMBlock, HighPass1D, MinibatchStdDev
from keras import ops


@keras.saving.register_keras_serializable()
class Critic(keras.Model):
    def __init__(self, time_dim, feature_dim, n_subjects, use_sublayer, use_channel_merger, **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.n_subjects = n_subjects
        self.use_sublayer = use_sublayer
        self.use_channel_merger = use_channel_merger
        self.input_shape = (time_dim, feature_dim)
        negative_slope = 0.1
        kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.d_sub = 32
        self.output_features = False

        self.sub_emb = torch.nn.Embedding(n_subjects, self.d_sub)
        if use_sublayer:
            self.sub_layer = SubjectLayers_FiLM(feature_dim, feature_dim, self.d_sub, init_id=True)

        if use_channel_merger:
            self.pos_emb = ChannelMerger(
                chout=feature_dim * 8, pos_dim=128, n_subjects=n_subjects, per_subject=True,
            )
            self.input_shape = (time_dim, feature_dim * 8)

        ks = 5

        self.post_att = keras.Sequential([
            keras.Input(shape=self.input_shape),
            LearnablePositionalEmbedding(512, 8),
            SelfAttention1D(2, 4)])
        
        self.film_block = FiLMBlock(8, 32)
        self.highpass = HighPass1D()
    
        self.conv1 = layers.Conv1D(feature_dim, ks, padding='same', name='conv3', kernel_initializer=kernel_initializer)
        self.act1  = layers.LeakyReLU(negative_slope=negative_slope)
        self.conv2 = layers.Conv1D(2 * feature_dim, ks, padding='same', name='conv4', kernel_initializer=kernel_initializer)
        self.pool2 = layers.AveragePooling1D(pool_size=2)
        self.act2  = layers.LeakyReLU(negative_slope=negative_slope)
        self.conv3 = layers.Conv1D(8 * feature_dim, ks, padding='same', name='conv5', kernel_initializer=kernel_initializer)
        self.pool3 = layers.AveragePooling1D(pool_size=2)
        self.act3  = layers.LeakyReLU(negative_slope=negative_slope)
        self.att2  = SelfAttention1D(8, feature_dim)
        self.conv4 = layers.Conv1D(16 * feature_dim, ks, padding='same', name='conv6', kernel_initializer=kernel_initializer)
        self.pool4 = layers.AveragePooling1D(pool_size=2)
        self.act4  = layers.LeakyReLU(negative_slope=negative_slope)
        self.flatten = layers.Flatten(name='dis_flatten')
        self.final_dense = layers.Dense(1, name='dis_dense6', dtype='float32', kernel_initializer=kernel_initializer)

        self.mbsdv = MinibatchStdDev()

        self.built = True  

    def call(self, inputs):
        x, sub_labels, positions = inputs['x'], inputs['sub'], inputs['pos']
        subj_emb = self.sub_emb(sub_labels.view(-1))
        if hasattr(self, 'sub_layer'):
            x = self.sub_layer(x, subj_emb)
        if hasattr(self, 'pos_emb'):
            x = self.pos_emb(x, sub_labels, positions)
        x = self.post_att(x)
        x = self.film_block(x, subj_emb)

        x_hp = self.highpass(x)        # (B, 512, 8), HF-emphasised
        x_cat = ops.concatenate([x, x_hp], axis=-1)  # (B, 512, 16)

        h1 = self.act1(self.conv1(x_cat))    # (B, 512, C1) HF-rich
        h  = self.act2(self.pool2(self.conv2(h1)))
        h  = self.act3(self.pool3(self.conv3(h)))
        h  = self.att2(h)
        h  = self.act4(self.pool4(self.conv4(h)))

        h = self.mbsdv(h)
        
        h_flat   = self.flatten(h)          # coarse features
        h1_flat  = self.flatten(h1)         # early HF features
        h_final = ops.concatenate([h_flat, h1_flat], axis=-1)

        if self.output_features:
            return h_final

        out = self.final_dense(h_final)
        return out
        
        # out = self.final_dense(h_final)
        # return out
    
    def extract_features(self, x, sub_labels, positions):
        subj_emb = self.sub_emb(ops.reshape(sub_labels, (-1,)))
        x = self.sub_layer(x, subj_emb)
        x = self.post_att(x)
        x = self.film_block(x, subj_emb)

        x_hp = self.highpass(x)
        x_cat = ops.concatenate([x, x_hp], axis=-1)

        h1 = self.act1(self.conv1(x_cat))
        h  = self.act2(self.pool2(self.conv2(h1)))
        h  = self.act3(self.pool3(self.conv3(h)))
        h  = self.att2(h)
        h  = self.act4(self.pool4(self.conv4(h)))

        # DCGAN-style pooling (global average here)
        pooled_h1 = ops.mean(h1, axis=1)
        pooled_h  = ops.mean(h,  axis=1)

        # you can also keep multiple stages if you like:
        feats = ops.concatenate([pooled_h1, pooled_h], axis=-1)
        return feats  # shape (B, D_feat)


    def get_config(self):
        config = super().get_config()
        config.update({
                      "time_dim": self.time_dim,
                      "feature_dim": self.feature_dim, 
                      "use_sublayer": self.use_sublayer,
                      "n_subjects": self.n_subjects,
                      "use_channel_merger": self.use_channel_merger})
        return config


@keras.saving.register_keras_serializable()
class Generator(keras.Model):
    def __init__(self, time_dim, feature_dim, latent_dim, use_sublayer,
                 n_subjects, use_channel_merger, interpolation, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.negative_slope = 0.2
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.use_sublayer = use_sublayer
        self.latent_dim = latent_dim
        self.n_subjects = n_subjects
        self.use_channel_merger = use_channel_merger
        self.interpolation = interpolation
        self.input_shape = (time_dim, feature_dim)
        kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.d_sub = 32
        self.sub_emb = torch.nn.Embedding(n_subjects, self.d_sub)

        if use_sublayer:
            self.sub_layer = SubjectLayers_FiLM(feature_dim, feature_dim, self.d_sub, init_id=True)

        if use_channel_merger:
            self.pos_emb = ChannelMerger(
                chout=feature_dim, pos_dim=32, n_subjects=n_subjects, per_subject=False,
            )

        self.post_att = keras.Sequential([
            keras.Input(shape=((latent_dim,))),
            layers.Dense(4096 * 1, kernel_initializer=kernel_initializer, name='gen_layer5'),
            layers.LeakyReLU(negative_slope=self.negative_slope, name='gen_layer6'),
            layers.Reshape((128, 32), name='gen_layer9'),
            LearnablePositionalEmbedding(128, 32),
            SelfAttention1D(4, 8)])
        
        self.film_block = FiLMBlock(32, 32)

        self.cov_block = keras.Sequential([
            keras.Input(shape=(128, 32)),
            *convBlock(filters=2 * [8 * feature_dim],
                       kernel_sizes= 2 * [3],
                       upsampling=[1, 1],
                       noiseinjection=[0, 0],
                       stride=1,
                       padding='same',
                       interpolation=interpolation,
                       negative_slope=0.2,
                       kernel_initializer=kernel_initializer,
                       batch_norm=True),
                       SelfAttention1D(4, 16),
                       layers.Conv1D(feature_dim, 3, padding='same', name='intermediate_conv', kernel_initializer=kernel_initializer),
                    #    layers.LeakyReLU(negative_slope=0.2),
        ], name='conv_block')

        self.dil_block = keras.Sequential([
            keras.Input(shape=(512, 8)),
            layers.Conv1D(feature_dim, 3, padding='same',
                            dilation_rate=2,
                            kernel_initializer=kernel_initializer,
                            name='dil_1_conv'),
            layers.LeakyReLU(negative_slope=0.2),
            layers.Conv1D(feature_dim, 3, padding='same',
                            dilation_rate=4,
                            kernel_initializer=kernel_initializer,
                            name='dil_2_conv'),
            layers.LeakyReLU(negative_slope=0.2),
        ], name="g_dilated_block")

        self.out_conv = layers.Conv1D(
                filters=feature_dim,
                kernel_size=5,        # or 7 for a stronger smoothing
                padding='same',
                activation=None,
                kernel_initializer=kernel_initializer,
                name='g_out_conv',
            )

        self.built = True

    def call(self, inputs):
        noise, sub_labels, positions = inputs
        x = self.post_att(noise)
        subj_emb = self.sub_emb(sub_labels.view(-1))
        x = self.film_block(x, subj_emb)
        x = self.cov_block(x)
        x = self.dil_block(x)
        x = self.out_conv(x)
        if hasattr(self, 'pos_emb'):
            x = self.pos_emb(x, sub_labels, positions)
        if hasattr(self, 'sub_layer'):
            x = self.sub_layer(x, subj_emb)
        if keras.mixed_precision.global_policy().name == 'mixed_float16':
            x = x.float()  # make sure the output is in float32 in mixed precision mode
        return x


    def get_config(self):
        config = super().get_config()
        config.update({
                      "time_dim": self.time_dim,
                      "feature_dim": self.feature_dim, 
                      "use_sublayer": self.use_sublayer,
                      "latent_dim": self.latent_dim,
                      "n_subjects": self.n_subjects,
                      "use_channel_merger": self.use_channel_merger,
                      "interpolation": self.interpolation})
        return config


@keras.saving.register_keras_serializable()
class FiLMGAN(keras.Model):
    def __init__(self,
                 time_dim=100, feature_dim=2, latent_dim=64, n_subjects=1,
                 use_sublayer_generator=False, use_sublayer_critic=False,
                 use_channel_merger_g=False,
                 use_channel_merger_c=False,
                 interpolation='bilinear',
                 steps_per_epoch = 500,
                 **kwargs):
        super().__init__(**kwargs)
        self.time_dim = time_dim
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.n_subjects = n_subjects
        self.use_sublayer_generator = use_sublayer_generator
        self.use_sublayer_critic = use_sublayer_critic
        self.use_channel_merger_g = use_channel_merger_g
        self.use_channel_merger_c = use_channel_merger_c
        self.interpolation = interpolation
        self.input_shape = (time_dim, feature_dim)
        self.d_loss_tracker = keras.metrics.Mean(name='d_loss')
        self.g_loss_tracker = keras.metrics.Mean(name='g_loss')
        self.accuracy_tracker = keras.metrics.BinaryAccuracy(name='accuracy')
        self.seed_generator = keras.random.SeedGenerator(42)

        # Training step counts
        self.global_step = 0        # counts train_step calls
        self.steps_per_epoch = steps_per_epoch  # Fix: our current setting!!
        self.warmup_epochs = 30

        self.generator = Generator(time_dim=time_dim,
                                   feature_dim=feature_dim,
                                   latent_dim=latent_dim,
                                   use_sublayer=use_sublayer_generator,
                                   n_subjects=n_subjects,
                                   use_channel_merger=use_channel_merger_g,
                                   interpolation=interpolation)

        self.critic = Critic(time_dim=time_dim,
                             feature_dim=feature_dim,
                             n_subjects=n_subjects,
                             use_sublayer=use_sublayer_critic,
                             use_channel_merger=use_channel_merger_c,)

        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker,
                self.accuracy_tracker]

    def get_config(self):
        config = super().get_config()
        config.update({
                      "time_dim": self.time_dim,
                      "feature_dim": self.feature_dim, 
                      "use_sublayer_generator": self.use_sublayer_generator,
                      "use_sublayer_critic": self.use_sublayer_critic,
                      "use_channel_merger_g": self.use_channel_merger_g,
                      "use_channel_merger_c": self.use_channel_merger_c,
                      "latent_dim": self.latent_dim,
                      "n_subjects": self.n_subjects,
                      "interpolation": self.interpolation})
        return config

    def call(self, x):
        return self.critic(x)

    def compile(self, d_optimizer, g_optimizer, gradient_penalty_weight):
        super().compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gradient_penalty_weight = gradient_penalty_weight

    def gradient_penalty(self, real_data, fake_data, sub, pos):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, device=real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        prob_interpolated = self.critic({'x': interpolated, 'sub': sub, 'pos': pos})

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size(), device=real_data.device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty
    

    def chk(self, name, t):
        if not torch.isfinite(t).all():
            print("NaNs at:", name, "max", t.abs().max().item())
            raise RuntimeError
        

    def train_step(self, data):
        real_data, sub, pos = data['x'], data['sub'], data['pos']

        device = next(self.parameters()).device
        real_data = real_data.to(device)
        sub = sub.to(device)
        pos = pos.to(device)

        batch_size = real_data.size(0)

        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        n_critic = 3 if self.global_step < warmup_steps else 1

        # train critic
        for _ in range(n_critic):
            noise = keras.random.normal((batch_size, self.latent_dim), dtype=real_data.dtype)
            perm = torch.randperm(batch_size, device=real_data.device)
            fake_sub = sub[perm].view(-1, 1)
            fake_data = self.generator((noise, fake_sub, pos)).detach() 
            real_pred = self.critic({'x': real_data, 'sub': sub, 'pos': pos})
            self.chk("D_real", real_pred)
            fake_pred = self.critic({'x': fake_data, 'sub': fake_sub, 'pos': pos})
            self.chk("D_fake", fake_pred)
            gp = self.gradient_penalty(real_data, fake_data.detach(), sub, pos)
            self.zero_grad()
            d_loss = (fake_pred.mean() - real_pred.mean()) + gp * self.gradient_penalty_weight
            d_loss.backward()

            grads = [v.value.grad for v in self.critic.trainable_weights]
            with torch.no_grad():
                self.d_optimizer.apply(grads, self.critic.trainable_weights)

        # Monitor gradient norms
        gradient_norms = []
        for p in self.critic.parameters():
            if p.grad is not None:
                gradient_norms.append(p.grad.norm().item())

        # train generator
        noise = keras.random.normal((batch_size, self.latent_dim), dtype=real_data.dtype)

        self.zero_grad()
        x_gen = self.generator((noise, fake_sub, pos))

        # smoking gun finbder 
        self.chk("G_out", x_gen)

        fake_pred = self.critic({'x': x_gen, 'sub': fake_sub, 'pos': pos})
        g_loss = -fake_pred.mean()
        g_loss.backward()

        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        total_loss = g_loss + d_loss

        # Update metrics and return their value
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        # Update globa step
        self.global_step += 1
        return {
            '1 d_loss': self.d_loss_tracker.result(),
            '2 g_loss': self.g_loss_tracker.result(),
            '3 critic_grad_norm': sum(gradient_norms) / len(gradient_norms),
            '4 gp': gp.item(),
            '5 real_pred': real_pred.mean().item(),
            '6 fake_pred': fake_pred.mean().item(),
            '7 real_pred_std': real_pred.std().item(),
            '8 fake_pred_std': fake_pred.std().item(),
            'loss': total_loss,
        }
