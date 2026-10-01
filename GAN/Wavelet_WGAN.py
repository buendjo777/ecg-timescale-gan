"""WGAN-GP

The generator maps a latent vector to a (64, seg_length, 2) scalogram and
the critic gives each scalogram a score.
A gradient penalty keeps the critic close to 1-Lipschitz.
The generator runs in mixed precision and the critic in float32.
seg_length is read from the data and only has to divide by 4.

Usage:
    CUDA_VISIBLE_DEVICES=0 python GAN/Wavelet_WGAN.py \\
        --data-dir /data/gan_afib_inverted --tag inverted --epochs 1000
    ... --resume        continues from the newest checkpoint

Writes sample PNGs, history.npy and the training checkpoints into a
results directory suffixed by --tag. The generator milestones and the
final models go to the working directory.
"""

import argparse
import os
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers, mixed_precision
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import GAN, SEED

for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as e:
        print(f"WARNING: memory growth not set on {_gpu.name}: {e}")


class WGAN:
    """Generator, critic and their optimisers."""

    def __init__(self, seg_length, n_scales=64, latent_dim=128, lr=5e-5, lambda_gp=10.0, n_critic=5, gen_updates=None, jit=True):
        self.seg_length = seg_length
        self.n_scales   = n_scales
        self.latent_dim = latent_dim
        self.lambda_gp  = float(lambda_gp)
        self.n_critic   = int(n_critic)

        self.img_shape = (n_scales, seg_length, 2)
        # Scales double three times (8 -> 64) but time only twice. The last upsample is (2, 1) and leaves time alone.
        assert seg_length % 4 == 0, f"seg_length {seg_length} must divide by 4"
        assert n_scales == 64, f"generator upsamples 8 to 64, got {n_scales}"
        self.init_w = seg_length // 4

        self.generator = self._build_generator()
        self.critic    = self._build_critic()
        # Adam uses betas (0, 0.9) and the rate follows a cosine decay
        # down to 5% of its starting value.
        def _sched(rate, steps):
            return (optimizers.schedules.CosineDecay(rate, steps, alpha=0.05) if steps else rate)
        self.g_opt = optimizers.Adam(_sched(lr, gen_updates), beta_1=0.0, beta_2=0.9)
        self.c_opt = optimizers.Adam(_sched(lr, gen_updates * self.n_critic if gen_updates else None), beta_1=0.0, beta_2=0.9)

        self.critic_step    = tf.function(self._critic_step, jit_compile=jit)
        self.generator_step = tf.function(self._generator_step, jit_compile=jit)

    def _build_generator(self):
        """Build the generator: latent vector -> scalogram in [-1, 1]."""
        return Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(8 * self.init_w * 128),
            layers.Reshape((8, self.init_w, 128)),
            layers.LayerNormalization(), layers.LeakyReLU(0.2),

            layers.UpSampling2D(), layers.Conv2D(128, 5, padding='same'),
            layers.LayerNormalization(), layers.LeakyReLU(0.2),

            layers.UpSampling2D(), layers.Conv2D(64, 5, padding='same'),
            layers.LayerNormalization(), layers.LeakyReLU(0.2),

            layers.UpSampling2D(size=(2, 1)),
            layers.Conv2D(32, 5, padding='same'),
            layers.LayerNormalization(), layers.LeakyReLU(0.2),

            layers.Conv2D(2, 3, padding='same', activation='tanh', dtype='float32'),
        ], name='Generator')

    def _build_critic(self):
        """Build the critic: scalogram -> unbounded score, all in float32."""
        f32 = {'dtype': 'float32'}
        return Sequential([
            layers.Input(shape=self.img_shape, dtype='float32'),
            layers.Conv2D(64,  5, strides=(2, 4), padding='same', **f32),
            layers.LayerNormalization(**f32),
            layers.LeakyReLU(0.2, **f32),
            layers.Conv2D(128, 5, strides=(2, 4), padding='same', **f32),
            layers.LayerNormalization(**f32),
            layers.LeakyReLU(0.2, **f32),
            layers.Conv2D(256, 5, strides=(2, 4), padding='same', **f32),
            layers.LayerNormalization(**f32),
            layers.LeakyReLU(0.2, **f32),
            layers.Flatten(**f32),
            layers.Dense(1, **f32),
        ], name='Critic')

    def gradient_penalty(self, real, fake):
        """Return (penalty, mean gradient norm) on real/fake interpolates."""
        alpha  = tf.random.uniform([tf.shape(real)[0], 1, 1, 1], 0.0, 1.0)
        interp = real + alpha * (fake - real)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.critic(interp, training=True)
        grads = tf.cast(tape.gradient(pred, interp), tf.float32)
        norm  = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2), tf.reduce_mean(norm)

    def _critic_step(self, real):
        """Run one critic update on one batch of real scalograms."""
        real  = tf.cast(real, tf.float32)
        noise = tf.random.normal([tf.shape(real)[0], self.latent_dim])
        fake = self.generator(noise, training=False)
        with tf.GradientTape() as tape:
            gp, gp_norm = self.gradient_penalty(real, fake)
            c_loss = (tf.reduce_mean(self.critic(fake, training=True)) - tf.reduce_mean(self.critic(real, training=True)) + self.lambda_gp * gp)
        self.c_opt.apply_gradients(zip(tape.gradient(c_loss, self.critic.trainable_variables), self.critic.trainable_variables))
        return c_loss, gp_norm

    def _generator_step(self, batch_size):
        """Run one generator update."""
        noise = tf.random.normal([batch_size, self.latent_dim])
        with tf.GradientTape() as tape:
            fake = self.generator(noise, training=True)
            g_loss = -tf.reduce_mean(self.critic(fake, training=True))
        self.g_opt.apply_gradients(zip(tape.gradient(g_loss, self.generator.trainable_variables), self.generator.trainable_variables))
        return g_loss


def save_sample(wgan, epoch, results_dir, fixed_noise):
    """Save the Re and Im channels generated from a fixed latent vector."""
    fake = wgan.generator(fixed_noise, training=False).numpy()[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, ch, title in zip(axes, [0, 1], ['Real', 'Imag']):
        ax.imshow(fake[:, :, ch], aspect='auto', cmap='RdBu_r', origin='lower')
        ax.set_title(f"Epoch {epoch} {title}")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/epoch_{epoch:04d}.png")
    plt.close()


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-dir',
                   default=os.path.dirname(GAN.scalogram_file) or '.',
                   help="directory holding processed_scalograms.npy")
    p.add_argument('--tag', default=None,
                   help="suffix for the results dir and checkpoints")
    p.add_argument('--seed', type=int, default=SEED,
                   help="RNG seed")
    p.add_argument('--sample-every', type=int, default=100,
                   help="epochs between sample PNGs")
    p.add_argument('--checkpoint-every', type=int, default=50,
                   help="epochs between checkpoints")
    p.add_argument('--epochs', type=int, default=1000,
                   help="epochs to train")
    p.add_argument('--resume', action='store_true',
                   help="continue from the newest checkpoint")
    p.add_argument('--no-xla', action='store_true',
                   help="disable XLA (jit_compile)")
    args = p.parse_args()

    mixed_precision.set_global_policy('mixed_bfloat16')
    print(f"precision: global={mixed_precision.global_policy().name}, critic=float32, xla={not args.no_xla}", flush=True)
    tf.keras.utils.set_random_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    results_dir = f"{GAN.results_dir}{suffix}"
    ckpt_stem = GAN.gen_model_file.replace('.keras', suffix)
    os.makedirs(results_dir, exist_ok=True)

    data = os.path.join(args.data_dir, os.path.basename(GAN.scalogram_file))
    # TODO: Change to streaming instead of copying data on RAM (unless RAM isnt an issue)
    X = np.load(data)
    n_scales, seg_length = X.shape[1], X.shape[2]
    print(f"Loaded {X.shape} from {data}", flush=True)

    batches_per_epoch = len(X) // GAN.batch_size
    gen_per_epoch = batches_per_epoch // GAN.n_critic
    if gen_per_epoch < 1:
        raise SystemExit(f"{batches_per_epoch} batches per epoch < n_critic={GAN.n_critic}: no generator update")
    total_gen = gen_per_epoch * args.epochs
    wgan = WGAN(seg_length=seg_length,
                n_scales=n_scales,
                latent_dim=GAN.latent_dim,
                lr=GAN.lr,
                lambda_gp=GAN.lambda_gp,
                n_critic=GAN.n_critic,
                gen_updates=total_gen,
                jit=not args.no_xla)
    print(f"{batches_per_epoch} batches/epoch, {gen_per_epoch} generator "
          f"updates/epoch, {args.epochs} epochs, {total_gen} generator "
          f"updates total ({total_gen * GAN.n_critic} critic updates)  "
          f"(g_lr={GAN.lr}, "
          f"seed={args.seed})", flush=True)

    # TODO: Shuffling indices might mix all the patients better, could better the GAN training.
    with tf.device('/CPU:0'):
        dataset = (tf.data.Dataset.from_tensor_slices(X)
                   .shuffle(min(len(X), 8192), seed=args.seed,
                            reshuffle_each_iteration=True)
                   .batch(GAN.batch_size, drop_remainder=True)
                   .prefetch(tf.data.AUTOTUNE))

    fixed_noise = tf.random.stateless_normal(
        [1, GAN.latent_dim], seed=[args.seed, 0])
    np.save(f"{results_dir}/fixed_noise.npy", fixed_noise.numpy())

    diag_batch = tf.cast(next(iter(dataset)), tf.float32)

    # Training state for --resume: both models and both optimisers. Only the newest two are kept.
    ckpt = tf.train.Checkpoint(generator=wgan.generator, critic=wgan.critic, g_opt=wgan.g_opt, c_opt=wgan.c_opt, epoch=tf.Variable(0, dtype=tf.int64))
    manager = tf.train.CheckpointManager(ckpt, f"{results_dir}/state", max_to_keep=2)

    start_epoch = 0
    if args.resume and manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        start_epoch = int(ckpt.epoch.numpy()) + 1
        print(f"resumed from {manager.latest_checkpoint} at epoch "
              f"{start_epoch}", flush=True)
    elif args.resume:
        print("--resume given but no checkpoint found, starting fresh",
              flush=True)

    # Each row is (epoch, c_loss, g_loss, d_real, d_fake, gp, gp_norm).
    history = []
    history_file = f"{results_dir}/history.npy"
    if start_epoch and os.path.isfile(history_file):
        history = [tuple(r) for r in np.load(history_file) if r[0] < start_epoch]
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        for i, batch in enumerate(dataset):
            c_loss, gp_norm = wgan.critic_step(batch)
            if i % GAN.n_critic == GAN.n_critic - 1:
                g_loss = wgan.generator_step(GAN.batch_size)

        if epoch % 10 == 0:
            d_real = tf.reduce_mean(wgan.critic(diag_batch, training=False))
            noise  = tf.random.normal([tf.shape(diag_batch)[0], wgan.latent_dim])
            fake   = wgan.generator(noise, training=False)
            d_fake = tf.reduce_mean(wgan.critic(fake, training=False))
            gp_val, gp_n = wgan.gradient_penalty(diag_batch, fake)
            print(f"Epoch {epoch:4d} | C {c_loss:.4f} | G {g_loss:.4f}"
                  f" | D_real {d_real:.4f} | D_fake {d_fake:.4f}"
                  f" | gap {float(d_real) - float(d_fake):8.4f}"
                  f" | GP {gp_val:.4f} | |grad| {gp_n:.4f}", flush=True)
            history.append((epoch, float(c_loss), float(g_loss), float(d_real), float(d_fake), float(gp_val), float(gp_n)))
            np.save(history_file, np.array(history))

        if epoch % args.sample_every == 0:
            save_sample(wgan, epoch, results_dir, fixed_noise)
        if epoch and epoch % args.checkpoint_every == 0:
            wgan.generator.save(f"{ckpt_stem}_ep{epoch}.keras")
            ckpt.epoch.assign(epoch)
            manager.save()

    elapsed = time.time() - t_start
    print(f"\n{args.epochs - start_epoch} epochs in {elapsed / 60:.1f} min", flush=True)

    save_sample(wgan, args.epochs, results_dir, fixed_noise)
    ckpt.epoch.assign(args.epochs - 1)
    manager.save()
    wgan.generator.save(f"{ckpt_stem}.keras")
    wgan.critic.save(f"{ckpt_stem}.keras".replace('wavelet_generator', 'wavelet_critic'))
    np.save(history_file, np.array(history))
    print(f"Done. Checkpoints at {ckpt_stem}*.keras, "
          f"samples in {results_dir}/", flush=True)


if __name__ == "__main__":
    main()
