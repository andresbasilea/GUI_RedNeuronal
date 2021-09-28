import numpy as np
from PIL import Image, ImageGrab, ImageOps
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    imagen = ImageGrab.grab(bbox=(900, 400, 1200, 700))
    #imagen_bn = ImageOps.grayscale(imagen)
    tamanio = 28,28
    imagen.thumbnail(tamanio, Image.ANTIALIAS)
    #imagen.show()
    #imagen.save(fp = "imagen1.jpg")
    #imagen_bn.thumbnail(tamanio,Image.ANTIALIAS) #Cambiar tamanio imagen
    NUMERO = hagamosMagia(imagen)
    # a 28 pixeles por 28 pixeles
    #arreglo = np.array(imagen_bn.getcolors())
    #print(arreglo)
    return NUMERO

def hagamosMagia(imagen): #TU CUERPO ES LA MAGIA BB
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    n_train = mnist.train.num_examples  # 55,000
    n_validation = mnist.validation.num_examples  # 5000
    n_test = mnist.test.num_examples  # 10,000
    n_input = 784  # primera capa red neuronal (28*28pi)
    n_hidden1 = 512  # primera capa oculta red neuronal
    n_hidden2 = 256
    n_hidden3 = 128
    n_output = 10  # capa de salida (digitos 0-9)
    learning_rate = 1e-4
    n_iterations = 2000  # cuantas veces pasamos por el paso de entrenamiento
    batch_size = 128  # cuantos ejemplos de entrenamiento usamos por cada paso
    dropout = 0.5  # prevenir overfitting

    # TENSOR -> estructura de datos similar a lista o arreglo
    # Se definen 3 tensores como placeholders (tensores a los que se les asignara valor despues)

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder(tf.float32)  # mismo tensor para entrenamiento que para testing

    weights = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }

    layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    layer_drop = tf.nn.dropout(layer_3, keep_prob)
    output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=Y, logits=output_layer
        ))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # train on mini batches
    for i in range(n_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={
            X: batch_x, Y: batch_y, keep_prob: dropout
        })

        # print loss and accuracy (per minibatch)
        if i % 100 == 0:
            minibatch_loss, minibatch_accuracy = sess.run(
                [cross_entropy, accuracy],
                feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
            )
            print(
                "Iteración # ",
                str(i),
                "\t| Pérdida =",
                str(minibatch_loss),
                "\t| Precisión =",
                str(minibatch_accuracy)
            )

    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
    print("\nPrecisión de set de prueba:", test_accuracy)

    img = np.invert(imagen.convert('L')).ravel()
    print(img)
    prediccion = sess.run(tf.argmax(output_layer, 1), feed_dict={X: [img]})
    prediccionF = np.squeeze(prediccion)
    print("Prediccion", np.squeeze(prediccion))
    return prediccionF