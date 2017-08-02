import os
import tensorflow as tf
from utils.config import *
from utils.batch import batch_iterator


class regressor:
    def __init__(self, input_dim, hidden_units, learning_rate):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        self.model_dir = self._str_model_dir()
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.sess = tf.Session(config=config)
        self._create_placeholders()
        self._create_loss()
        self._create_optimizer()

        self.summ = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.model_dir)
        self.writer.add_graph(self.sess.graph)

        try:
            self.saver.restore(self.sess, os.path.join(self.model_dir, "model.ckpt"))
        except:
            print(
                "Can't find any existing checkpoint. "
                "Creating a new model that needs to be trained."
                )
            self.sess.run(tf.global_variables_initializer())

    def train(self, training_set, test_set, label, batch_size, n_epoch, write_metrics):
        print()
        print('Starting run for %s' % self.model_dir)
        for i in range(n_epoch):
            for batch in batch_iterator(training_set, batch_size):
                self._training_step(i, i % write_metrics == 0, batch, test_set, label)
            print()
            print("epoch %s out of %s" % (str(i + 1), str(n_epoch)))

        save_path = self.saver.save(self.sess, os.path.join(self.model_dir, "model.ckpt"))
        print("Model saved in file %s" % save_path)
        print('Run `tensorboard --logdir=%s` to see the results.' % MODELS_DIR)
        print()

    def predict(self, dataset, label):
        predictions = self.sess.run(
            self.Y_pred,
            {
                self.x: dataset.drop([label], axis=1).as_matrix(),
                self.y:  dataset[label].as_matrix()
            }
        )
        return predictions

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.input_dim],
                                name="x")
        self.y = tf.placeholder(tf.float32,
                                shape=[None, ],
                                name="y")

    def _create_loss(self):
        """ Step 3 + 4: define the inference + the loss function """
        layer_out, relu_out = self._stack_layers(self.x,
                                                 self.input_dim,
                                                 self.hidden_units)
        self.Y_pred = self._nn_layer(relu_out,
                                     layer_out,
                                     1,
                                     'fc_out',
                                     act=tf.identity)
        self.Y_pred = tf.reshape(self.Y_pred, [-1])
        with tf.name_scope("rmsle"):
            mean_log = tf.reduce_mean(tf.square(self.y-self.Y_pred))
            self.RMSLE = tf.sqrt(mean_log, name="rmsle")
            tf.summary.scalar("rmsle", self.RMSLE)

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.name_scope("train"):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate)\
                .minimize(self.RMSLE)

    def _stack_layers(self, start_input, dim_in, hidden_units):
        hidden_units.insert(0, dim_in)
        next_input = start_input
        for layer_nb in range(len(hidden_units) - 2):
            layer_in = hidden_units[layer_nb]
            layer_out = hidden_units[layer_nb + 1]
            relu = self._nn_layer(next_input, layer_in, layer_out, "fc" + str(layer_nb))
            next_input = relu
        return layer_out, next_input

    def _nn_layer(self, input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.

        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self._weight_variable([input_dim, output_dim])
                self._variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self._bias_variable([output_dim])
                self._variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def _str_model_dir(self):
        subdir = "hidden_units " + str(self.hidden_units)
        subdir += " learning_rate " + str(self.learning_rate)
        model_dir = "/".join([MODELS_DIR, subdir])
        return model_dir

    def _training_step(self, i, write_metrics, training_set, test_set, label):
        if write_metrics:
            s = self.sess.run(
                self.summ,
                {
                    self.x: test_set.drop([label], axis=1).as_matrix(),
                    self.y:  test_set[label].as_matrix()
                }
            )
            self.writer.add_summary(s, i)
        self.sess.run(
            self.train_step,
            {
                self.x: training_set.drop([label], axis=1).as_matrix(),
                self.y:  training_set[label].as_matrix()
            }
        )
