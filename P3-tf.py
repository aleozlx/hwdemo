import os
from tqdm import tqdm
import tensorflow as tf

epochs = 50
batch_size = 500
steps_per_epoch = 5000//batch_size
summary = True

def mnist_reader(part):
    for i in range(10):
        with open('Part3_{}_{}.csv'.format(i, part)) as f:
            for line in f:
                yield line.strip(), i

def mnist_decoder(csv_line, label):
    FIELD_DEFAULTS = [[0.0]]*(28*28)
    with tf.variable_scope('DataSource'):
        fields = tf.decode_csv(csv_line, FIELD_DEFAULTS)
        im = tf.stack(fields)
        im = tf.reshape(im, (28, 28, 1))
        return im, tf.one_hot(label, depth=10)

tf.reset_default_graph()

with tf.variable_scope('DataSource'):
    dataset = tf.data.Dataset.from_generator(lambda: mnist_reader('Train'),
        # csv_line, label
        (tf.string, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([])
    )).map(mnist_decoder, num_parallel_calls=2) \
      .shuffle(5000) \
      .batch(batch_size) \
      .prefetch(1) \
      .repeat(epochs)

    dataset_val = tf.data.Dataset.from_generator(lambda: mnist_reader('Test'),
        (tf.string, tf.int32),
        (tf.TensorShape([]), tf.TensorShape([])
    )).map(mnist_decoder, num_parallel_calls=2) \
      .batch(batch_size) \
      .prefetch(1)

    iter_handle = tf.placeholder(tf.string, shape=[])
    data_iterator = tf.data.Iterator.from_string_handle(iter_handle, dataset.output_types, dataset.output_shapes)
    train_iterator = dataset.make_one_shot_iterator()
    val_iterator = dataset_val.make_initializable_iterator()
    val_init_op = data_iterator.make_initializer(dataset_val)
    images, onehot_labels = data_iterator.get_next()

with tf.variable_scope('CNN'):
    convmaps = tf.keras.layers.Conv2D(16, (7,7), activation='tanh')(images)
    features = tf.reshape(convmaps, (batch_size, 16*22*22))
    fc1 = tf.keras.layers.Dense(128, activation='tanh')(features)
    pred = tf.keras.layers.Dense(10)(features)

with tf.variable_scope('LossFn'):
   total_loss = tf.nn.softmax_cross_entropy_with_logits_v2(onehot_labels, pred)

with tf.variable_scope('Optimizer'):
    global_step = tf.Variable(0)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(total_loss, global_step)

with tf.variable_scope('Metrics'):
    labels = tf.to_int32(tf.argmax(onehot_labels, axis=1))
    argmax_idx = tf.to_int32(tf.argmax(pred, axis=1))
    epoch_loss_avg, epoch_loss_avg_update = tf.metrics.mean(total_loss)
    epoch_accuracy, epoch_accuracy_update = tf.metrics.accuracy(labels, argmax_idx)
    if summary:
        summary_loss = tf.summary.scalar("loss", epoch_loss_avg)
        summary_vloss = tf.summary.scalar("vloss", epoch_loss_avg)
        summary_acc = tf.summary.scalar("acc", epoch_accuracy)
        summary_vacc = tf.summary.scalar("vacc", epoch_accuracy)
        summary_train = tf.summary.merge([summary_loss, summary_acc])
        summary_val = tf.summary.merge([summary_vloss, summary_vacc])

if summary:
    run_dir = os.path.join('/tmp/logdir', 'cnn-mlp')
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

with tf.variable_scope('Initializer'):
    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

config = tf.ConfigProto(
    intra_op_parallelism_threads=2,
    inter_op_parallelism_threads=2,
    allow_soft_placement=True)

with tf.Session(config = config) as sess:
    sess.run(init_global)
    train_handle = sess.run(train_iterator.string_handle())
    val_handle = sess.run(val_iterator.string_handle())

    if summary:
        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)

    for epoch in range(epochs):
        sess.run(init_local)
        for step in tqdm(range(steps_per_epoch)):
            sess.run([train_op,
                      total_loss,
                      epoch_loss_avg_update,
                      epoch_accuracy_update],
                     feed_dict={iter_handle: train_handle})
            if summary:
                summary = sess.run(summary_train)
        if summary:
            summary_writer.add_summary(summary, epoch)
        print('epoch', epoch+1, 'acc', sess.run(epoch_accuracy), end=' ')

        sess.run([init_local, val_init_op], feed_dict={iter_handle: val_handle})
        for val_step in range(steps_per_epoch):
            sess.run([total_loss,
                      epoch_loss_avg_update,
                      epoch_accuracy_update],
                     feed_dict={iter_handle: val_handle})
            if summary:
                summary = sess.run(summary_val)
        if summary:
            summary_writer.add_summary(summary, epoch)
        print('vacc', sess.run(epoch_accuracy))

