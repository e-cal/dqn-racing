import tensorflow as tf

# install tensorflow-gpu==2.11.0 instead of tensorflow to use gpu
# also need cuda [sys] and tenorrt [pip]

# if script can't find libdevices, symlink cuda install dir to /usr/local
# or export XLA_FLAGS=--xla_gpu_cuda_data_dir=<PATH_TO_CUDA>

gpus = tf.config.experimental.list_physical_devices("GPU")

print("Num GPUs Available: ", len(gpus))
