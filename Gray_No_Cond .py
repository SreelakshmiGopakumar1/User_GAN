import tensorflow as tf
from utils import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import sys, os, glob
from tqdm import tqdm,trange
from tqdm._utils import _term_move_up
import subprocess as sp
import csv
import pandas as pd

# height, width and number of channels of the images (grayscale)
HEIGHT, WIDTH, CHANNEL = 100, 100, 1
#size of the kernel for the convolution
KERNEL_SIZE = 5
# size of the batch for the training
BATCH_SIZE = 64
# number of epochs
EPOCH = 1
# EPOCH = 10
# number of images in the training set
TRAINING_SET_SIZE = 70
# dimension of the input noise vector
Z_dim = 26
# dimension of the vector of labels
y_dim = 14
# multiplication factor for the training set augmentation
MUL_FACTOR = 12
# learning rate for the gradient descent optimization algorithm
LEARNING_RATE = 1e-4
# number of times the discriminator is updated for every batch
D_ITER = 5
# number of times the generator is updated for every batch
G_ITER = 1

COLUMNS_ = ["No.","Labels","Rating"]

CSV_DATA = []

CSV_FILE = "rating_input.csv"

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth=True

# ReLU activation function
def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)


# uniform noise generation
def sample_Z(m):
    return np.random.uniform(-1., 1., size=[m])

def save_image(images, label, path):
    """
    Saves each generated images in a folder along with the label
    """
    if not os.path.isdir(path):os.makedirs(path)
    label_def = [
        'Coast',
        'Sea/Ocean',
        'Mountain',
        'Wood/Trees',
        'Plain',
        'Sand/Desert',
        'Isle',
        'Valley',
        'Rock/Stone',
        'Hills',
        'Grass',
        'Lake',
        'Promontory',
        'Gulf'
    ]
    for i, im in enumerate(images):
        print("saving image(%d/%d)"%(i,len(images)),end="\r")
        im = (255*(im + 1)/2).astype(np.uint8)
        y = np.argwhere(label[i]>0).squeeze() ### this has to be modified when the conditioning information is available
        title = ', '.join([label_def[i] for i in y])
        CSV_DATA.append({'No.':i,"Labels":title,"Rating":0})
        plt.imshow(im.squeeze(), cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(path + '/%d.png'%(i), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

# read the labels from the training set txt file and to generate the noise corresponding to the images
def read_labels():
    print("Reading Labels...")
    file = open('Training_set.txt', 'r')
    lines = file.readlines()
    file.close()
    objects = []
    noises = []
    for i, line in enumerate(lines):
        if i < len(lines):
            if " " in line:
                param, value = line.split(" ", 1)
                obj_float=np.fromstring(value, dtype=float, sep=",")

                if len(obj_float) != y_dim:
                    sys.exit("Error while reading the lables!")

                # replicate the labels by MUL_FACTOR, because the images will be replicated as well
                for j in range(MUL_FACTOR):
                    random_label = sample_Z(y_dim)
                    objects.append(random_label)
                    #objects.append(obj_float)
                    input_noise = sample_Z(Z_dim)
                    noises.append(input_noise)

    tr_labels = tf.stack(objects)
    tr_noises = tf.stack(noises)
    return tr_labels, tr_noises


# read the MODIS images from the file, perform the augmentation and create the batches
def process_data(images_path):
    cwd = os.getcwd()
    os.chdir(images_path)
    image_names = []
    for root, dirs, files in os.walk("."):
        for filename in files:
            if filename=='1.png':
                filename_mod = str("%s/%s"%(root,filename)).replace("./","")
                image_names.append(filename_mod)
    i = 0
    images = []
    print("Augmenting training Data...")
    for im in tqdm(image_names,total=len(image_names)):
        if im!='.DS_Store':
            I1 = np.array(Image.open(im).convert('L'))
            # perform 3 rotations
            I2 = np.rot90(I1)
            I3 = np.rot90(I1, 2)
            I4 = np.rot90(I1, 3)
            # perform 2 flips
            I5 = np.flipud(I1)
            I6 = np.fliplr(I1)
            I7 = np.flipud(I2)
            I8 = np.fliplr(I2)
            I9 = np.flipud(I3)
            I10 = np.fliplr(I4)
            I11 = np.flipud(I4)
            I12 = np.fliplr(I4)
            images.append(I1)
            images.append(I2)
            images.append(I3)
            images.append(I4)
            images.append(I5)
            images.append(I6)
            images.append(I7)
            images.append(I8)
            images.append(I9)
            images.append(I10)
            images.append(I11)
            images.append(I12)
            i = i + MUL_FACTOR

    real_images = []
    print("Manipulating fetched images...")
    for im in tqdm(images,total=len(images)):
    	# convert into float
        real_image = tf.cast(im, tf.float32)
        real_image = (2*real_image/255.0)-1
        real_images.append(real_image)

    train_images = tf.stack(real_images)
    train_images2 = train_images[0:10]

    print("Length of images: " + str(len(images)))

    train_images = tf.expand_dims(train_images, 3)
    print("Shape of training set(expanded): " + str(train_images.get_shape()))

    # #os.chdir('C:\\Users\\tonin\\Desktop\\Materiale da consegnare\\Code\\')
    os.chdir('../../Code')
    train_labels, train_noises = read_labels()
    train_labels2, train_noises2 = train_labels[0:10], train_noises[0:10]
    # print(len(train_images),len(train_labels),len(train_noises))

    train_noises.set_shape([MUL_FACTOR*TRAINING_SET_SIZE, Z_dim])

    # create the batch with real images and corresponding labels
    random_image, random_label, random_noise = tf.train.slice_input_producer([train_images, train_labels, train_noises], shuffle=True)
    images_batch, labels_batch, noise_batch = tf.train.batch([random_image, random_label, random_noise], batch_size=BATCH_SIZE)

    # create the batch with fake images and corresponding labels
    fake_images, fake_labels = tf.train.slice_input_producer([train_images,train_labels,], shuffle=True)
    fake_images_batch, fake_labels_batch = tf.train.batch([fake_images, fake_labels], batch_size=BATCH_SIZE)

    num_images = i

    print("Shape of images_batch" + str(images_batch.get_shape()))
    print("Shape of fake_images_batch" + str(fake_images_batch.get_shape()))
    print("Shape of labels_batch"+ str(labels_batch.get_shape()))
    os.chdir(cwd)
    return images_batch, fake_images_batch, labels_batch, noise_batch,  num_images


# generator function
def generator(z, y, is_train, reuse=False):

    #number of filters used for the convolutions
    c7, c14, c28, c56 = 64, 32, 16, 8
    s7 = 7
    output_dim=CHANNEL

    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[Z_dim+y_dim, s7 * s7 * c7], dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c7 * s7 * s7], dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0))
        # concatenation between noise and labels
        conc=tf.concat(axis=1, values=[z, y])
        # fully connected layer, outputting 7*7 images
        # 7*7*64
        flat_conv1 = tf.add(tf.matmul(conc, w1), b1, name='flat_conv1')
        # convolution, batch normalization, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s7, s7, c7], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None,scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 14*14*32
        conv2 = tf.layers.conv2d_transpose(act1, c14, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act3')
        # 28*28*16
        conv3 = tf.layers.conv2d_transpose(act2, c28, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2],padding="SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None,scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 56*56*8
        conv4 = tf.layers.conv2d_transpose(act3, c56, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2],padding="SAME",
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02), name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay=0.9, updates_collections=None,
                                   scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')

        # 112*112*1
        conv5 = tf.layers.conv2d_transpose(act4, output_dim, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),name='conv5')
        # activation function
        act5 = tf.nn.tanh(conv5, name='act5')

        # the edges are cut to reach the desired 100*100 shape
        # 100*100*1
        act5 = act5[:, 6:106, 6:106, :]

        return act5


# discriminator function
def discriminator(x, y, is_train, reuse=False):

    #number of filters used for the convolutions
    c100, c50, c25 = 8, 16, 32
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        x = tf.reshape(x, shape=[BATCH_SIZE, HEIGHT, WIDTH, CHANNEL])
        # Convolution, batch normalization, activation, repeat!
        # 50*50*8
        conv1 = tf.layers.conv2d(x, c100, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn1')
        act1 = lrelu(bn1, n='act1')
        # 25*25*16
        conv2 = tf.layers.conv2d(act1, c50, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay=0.9,
                                           updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        # 13*13*32
        conv3 = tf.layers.conv2d(act2, c25, kernel_size=[KERNEL_SIZE, KERNEL_SIZE], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay=0.9,
                                       updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')

        dim = int(np.prod(act3.get_shape()[1:]))
        # images reshaped into vectors
        fc1 = tf.reshape(act3, shape=[-1, dim], name='fc1')
        # first fc layer
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 100], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[100], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        fc2 = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # concatenation with the labels
        conc = tf.concat(axis=1, values=[fc2, y])
        # second fc layer
        w3 = tf.get_variable('w3', shape=[conc.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        b3 = tf.get_variable('b3', shape=[1], dtype=tf.float32,
                         initializer=tf.constant_initializer(0.0))
        logits = tf.add(tf.matmul(conc, w3), b3, name='logits')

        print(logits)

        return logits


# function to plot the samples in a grid
def plot(samples):

    fig = plt.figure()
    gs = gridspec.GridSpec(8,8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(HEIGHT, WIDTH), cmap='Greys_r')

    plt.show()

    return fig


# train function
def train(data_dir, save_dir, user_feedback, restore_dir=None):

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL])
        X_fake = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, CHANNEL])
        Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        y = tf.placeholder(tf.float32, shape=[None, y_dim])
        is_train = tf.placeholder(tf.bool, name='is_train')
        user_eval = tf.placeholder(tf.float32, shape=(), name='user_evaluation')

    # creation of the folder for the results
    k = 1
    while (os.path.exists('out_labels'+str(k)+'/')):
        k+=1
    # os.makedirs('out_labels'+str(k)+'/')
    version = 'GAN-'+str(k)

    start_time = time.time()

    # definition of the generator and of the discriminators
    G_sample = generator(Z, y, is_train)
    D_real = discriminator(X, y, is_train)
    D_wrong = discriminator(X_fake, y, is_train, reuse=True)
    D_fake = discriminator(G_sample, y, is_train, reuse=True)

    # Discriminator loss and generator loss
    D_loss = (tf.reduce_mean(D_fake) + tf.reduce_mean(D_wrong))/2 - tf.reduce_mean(D_real)  # This optimizes the discriminator.
    G_loss = -user_eval*tf.reduce_mean(D_fake)  # This optimizes the generator.

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]

    # RMSprop optimization algorithm used on both discriminator and generator
    D_solver = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(D_loss, var_list=d_vars)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(G_loss, var_list=g_vars)

    # weight clipping (Wasserstein GAN)
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]

    batch_size = BATCH_SIZE
    image_batch, fake_batch, label_batch, noise_batch, samples_num = process_data(data_dir)
    batch_num = int(samples_num / batch_size)

    saver = tf.train.Saver()
    model_dir = save_dir + '/checkpoint/model'
    if not os.path.isdir(save_dir + '/checkpoint'):
        print('creating path')
        os.mkdir(save_dir + '/checkpoint')

    dlosses = []
    glosses = []
    init_op = tf.global_variables_initializer()
    print('TRAINING STARTED...')
    # RATING = 0
    # train for the correct number of epochs
    with tf.Session(config=gpu_config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ## If restore directory is given initialize the parameters from a check point else initialize randomly
        if not restore_dir is None:
            saver.restore(sess, restore_dir)
            xl_data = pd.read_csv(save_dir+'/rating_input.csv',skipinitialspace=True)
            user_feedback = np.average(data.Rating.tolist())

        else:
            sess.run(init_op)

        tmp_clear = sp.call('clear',shell=True)
        for it in range(EPOCH):
            print('TRAINING...\n')
            print("EPOCH %d/%d:\n"%(it+1,EPOCH))
            ### reset user feedback after 10 iterations#####
            if it > 10:
                user_feedback = 1.0
            # for every epoch use multiple minibatches
            for i in range(batch_num):
                print("Batch %d/%d:"%(i+1,batch_num))
                # the discriminator is trained more time than the generator
                for j in tqdm(range(D_ITER)):
                    train_image, fake_train_image, train_labels, train_noise = sess.run([image_batch, fake_batch, label_batch, noise_batch])
                    train_labels = train_labels.astype(np.float32)
                    # the discriminator loss is computed and the weights are updated. Then the weights are clipped.
                    _, D_loss_curr, _ = sess.run([D_solver, D_loss, d_clip], feed_dict={X: train_image, X_fake: fake_train_image, Z: train_noise, y: train_labels, is_train:True })
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[F")


                for j in range (G_ITER):
                    # the generator loss is computed and the weights are updated.
                    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: train_noise, y: train_labels, is_train:True, user_eval:user_feedback})
            tmp_clear = sp.call('clear',shell=True)

            dlosses.append(D_loss_curr)
            glosses.append(G_loss_curr)

            # # Results saved every 100 epochs
            if (it+1) % 10 == 0:
                # saving of the images
                samples = sess.run(G_sample, feed_dict={Z: train_noise, y: train_labels, is_train: False})
                save_images(samples, [8, 8], 'out_labels' + '/'+str(it).zfill(3)+'.png')
        ## generate images and save them in a folder
        samples = sess.run(G_sample, feed_dict={Z: train_noise, y: train_labels, is_train: False})
        save_image(samples, train_labels, save_dir + '/images')
        try:
            with open(save_dir + '/' + CSV_FILE, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=COLUMNS_)
                writer.writeheader()
                for data in CSV_DATA:
                    writer.writerow(data)
        except IOError:
            print("I/O error")
        coord.request_stop()
        coord.join(threads)
        saver.save(sess, model_dir)
        print('Training finished!')


# test function: no discriminator needed
def test():
    print("Test:")
    # placeholders
    with tf.variable_scope('input'):

        Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
        y = tf.placeholder(tf.float32, shape=[None, y_dim])
        is_train = tf.placeholder(tf.bool, name='is_train')

    # generator
    G_sample = generator(Z, y, is_train)

    # select the correct version of the GAN for grayscales
    version = 'GAN-gray'
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    variables_to_restore = slim.get_variables_to_restore(include=['gen'])
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.latest_checkpoint('./Model/' + version)
    saver.restore(sess, ckpt)

    test_noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, Z_dim]).astype(np.float32)
    #test_labels = np.empty((0, y_dim))
    test_labels = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, y_dim]).astype(np.float32)

    # images generation
    samples = sess.run(G_sample, feed_dict={Z: test_noise, y: test_labels, is_train: False})

    # saving of the results
    save_images(samples, [8, 8], 'out_labels_test/Gray_no_Cond_out.png')



if __name__ == "__main__":
    print("initialized Training session...")
    train()
    # test()
