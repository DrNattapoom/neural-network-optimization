import time
import tensorflow as tf
import matplotlib.pyplot as plt


def partition(
    dataset: tf.data.Dataset, train: float = 0.8, test: float = 0.1, val: float = 0.1,
    shuffle: bool = True, shuffle_size: int = 1000, seed: int = 101
) -> tuple:
    assert(train + test + val == 1)

    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed)

    dataset_size = len(dataset)
    train_size = int(train * dataset_size)
    val_size = int(val * dataset_size)

    dataset_train = dataset.take(train_size)
    dataset_test = dataset.skip(train_size).skip(val_size)
    dataset_val = dataset.skip(train_size).take(val_size)

    return dataset_train, dataset_test, dataset_val


def plot(sgd_history, adam_history, key: str = 'accuracy'):
    plt.plot(sgd_history.history[key], label='SGD')
    plt.plot(adam_history.history[key], label='Adam')
    plt.title(f'Model {key.title()}')
    plt.xlabel('Epoch')
    plt.ylabel(key.title())
    plt.legend()
    plt.show()


def collect_data_titanic(
    titanic: tf.data.Dataset, batch_sizes: list[int], learning_rates: list[int],
    optimizers: list[tf.keras.optimizers.Optimizer]
) -> list:
    data = []

    fig, axs = plt.subplots(len(batch_sizes), len(
        learning_rates), figsize=(20, 20))
    fig.suptitle("Binary Classification: Model Accuracy")

    for i, batch_size in enumerate(batch_sizes):
        for j, learning_rate in enumerate(learning_rates):
            titanic_train, titanic_test, titanic_val = partition(titanic)
            AUTOTUNE = tf.data.experimental.AUTOTUNE

            def handle_data(data, label):
                age = data['age']
                sex = data['sex']
                pclass = data['pclass']
                fare = data['fare']
                handled_data = tf.stack([
                    tf.cast(age, tf.int64),
                    sex,
                    pclass,
                    tf.cast(fare, tf.int64)
                ])
                return handled_data, label

            def process_titanic(titanic):
                titanic = titanic.map(handle_data, num_parallel_calls=AUTOTUNE)
                titanic = titanic.shuffle(len(titanic))
                titanic = titanic.batch(batch_size)
                titanic = titanic.prefetch(AUTOTUNE)
                return titanic
            
            titanic_train = process_titanic(titanic_train)
            titanic_test = process_titanic(titanic_test)
            titanic_val = process_titanic(titanic_val)

            model = tf.keras.Sequential([
                tf.keras.layers.Input((4, )),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])

            for optimizer in optimizers:
                optimizer_name = str(optimizer).split(".")[-2]
                print("====================")
                print(
                    f"{optimizer_name}: batch_size = {batch_size}; learning_rate = {learning_rate}")
                print("====================")
                model.compile(
                    optimizer=optimizer(learning_rate=learning_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy']
                )
                start_time = time.time()
                history = model.fit(
                    titanic_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=titanic_val,
                    verbose=0
                )
                training_time = time.time() - start_time
                results = model.evaluate(titanic_test, verbose=0)
                print("--------------------")
                print(f"loss: {results[0]}, accuracy: {results[1]}")
                print("--------------------")
                axs[i, j].plot(history.history['accuracy'],
                               label=optimizer_name)
                data.append(('titanic', batch_size, learning_rate, optimizer_name,
                            training_time, results[0], results[1]))

            axs[i, j].set_title(
                f"(batch size = {batch_size}; learning rate = {learning_rate})", fontsize=10)
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel('Accuracy')
            axs[i, j].legend()

    fig.savefig('titanic_full_figure.png')

    return data


def collect_data_mnist(
    mnist: tf.data.Dataset, batch_sizes: list[int], learning_rates: list[int],
    optimizers: list[tf.keras.optimizers.Optimizer]
) -> list:
    data = []

    fig, axs = plt.subplots(len(batch_sizes), len(
        learning_rates), figsize=(20, 20))
    fig.suptitle("Multi-class Image Classification: Model Accuracy")

    for i, batch_size in enumerate(batch_sizes):
        for j, learning_rate in enumerate(learning_rates):
            mnist_train, mnist_test, mnist_val = partition(mnist)

            AUTOTUNE = tf.data.experimental.AUTOTUNE

            def normalize_image(image, label):
                return tf.cast(image, tf.float32) / 255, label

            def process_mnist(mnist):
                mnist = mnist.map(normalize_image, num_parallel_calls=AUTOTUNE)
                mnist = mnist.shuffle(len(mnist))
                mnist = mnist.batch(batch_size)
                mnist = mnist.prefetch(AUTOTUNE)
                return mnist

            mnist_train = process_mnist(mnist_train)
            mnist_test = process_mnist(mnist_test)
            mnist_val = process_mnist(mnist_val)

            model = tf.keras.models.Sequential([
                tf.keras.Input((28, 28, 1)),
                tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, 3, activation = 'relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation = 'relu'),
                tf.keras.layers.Dense(10)
            ])

            for optimizer in optimizers:
                optimizer_name = str(optimizer).split(".")[-2]
                print("====================")
                print(
                    f"{optimizer_name}: batch_size = {batch_size}; learning_rate = {learning_rate}")
                print("====================")
                model.compile(
                    optimizer = optimizer(learning_rate = learning_rate),
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                    metrics = ['accuracy']
                )
                start_time = time.time()
                history = model.fit(
                    mnist_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=mnist_val,
                    verbose=0
                )
                training_time = time.time() - start_time
                results = model.evaluate(mnist_test, verbose=0)
                print("--------------------")
                print(f"loss: {results[0]}, accuracy: {results[1]}")
                print("--------------------")
                axs[i, j].plot(history.history['accuracy'],
                               label=optimizer_name)
                data.append(('mnist', batch_size, learning_rate, optimizer_name,
                            training_time, results[0], results[1]))

            axs[i, j].set_title(
                f"(batch size = {batch_size}; learning rate = {learning_rate})", fontsize=10)
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel('Accuracy')
            axs[i, j].legend()

    fig.savefig('mnist_full_figure.png')

    return data

def collect_data_imdb(
    imdb: tf.data.Dataset, batch_sizes: list[int], learning_rates: list[int],
    optimizers: list[tf.keras.optimizers.Optimizer]
) -> list:
    data = []

    fig, axs = plt.subplots(len(batch_sizes), len(
        learning_rates), figsize=(20, 20))
    fig.suptitle("Sentimentaal Analysis: Model Accuracy")

    for i, batch_size in enumerate(batch_sizes):
        for j, learning_rate in enumerate(learning_rates):
            imdb_train, imdb_test, imdb_val = partition(imdb)

            AUTOTUNE = tf.data.experimental.AUTOTUNE

            encoder = tf.keras.layers.TextVectorization()
            encoder.adapt(imdb_train.map(lambda text, _: text))

            def encode(text, label):
                return encoder(text), label

            def process_imdb(imdb):
                imdb = imdb.map(encode, num_parallel_calls = AUTOTUNE)
                imdb = imdb.shuffle(len(imdb))
                imdb = imdb.padded_batch(batch_size, padded_shapes = ([None], ()))
                imdb = imdb.prefetch(AUTOTUNE)
                return imdb

            imdb_train = process_imdb(imdb_train)
            imdb_test = process_imdb(imdb_test)
            imdb_val = process_imdb(imdb_val)

            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero = True),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                tf.keras.layers.Dense(64, activation = "relu"),
                tf.keras.layers.Dense(1)
            ])

            for optimizer in optimizers:
                optimizer_name = str(optimizer).split(".")[-2]
                print("====================")
                print(
                    f"{optimizer_name}: batch_size = {batch_size}; learning_rate = {learning_rate}")
                print("====================")
                model.compile(
                    optimizer = optimizer(learning_rate = learning_rate, clipnorm = 1),
                    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                    metrics = ['accuracy']
                )
                start_time = time.time()
                history = model.fit(
                    imdb_train,
                    epochs=10,
                    batch_size=batch_size,
                    validation_data=imdb_val,
                    verbose=0
                )
                training_time = time.time() - start_time
                results = model.evaluate(imdb_test, verbose=0)
                print("--------------------")
                print(f"loss: {results[0]}, accuracy: {results[1]}")
                print("--------------------")
                axs[i, j].plot(history.history['accuracy'],
                               label=optimizer_name)
                data.append(('imdb', batch_size, learning_rate, optimizer_name,
                            training_time, results[0], results[1]))

            axs[i, j].set_title(
                f"(batch size = {batch_size}; learning rate = {learning_rate})", fontsize=10)
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel('Accuracy')
            axs[i, j].legend()

    fig.savefig('imdb_full_figure.png')

    return data
