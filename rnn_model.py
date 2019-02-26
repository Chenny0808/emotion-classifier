from tensorflow import keras
import numpy as np
import tensorflow as tf


tokenizer = keras.preprocessing.text.Tokenizer()


def dataset_preparation(train_data_path, valid_data_path, test_data_path):
    with open(train_data_path, 'r', encoding="utf-8") as f:
        train_data = f.readlines()

    with open(valid_data_path, 'r', encoding="utf-8") as f:
        valid_data = f.readlines()

    with open(test_data_path, 'r', encoding="utf-8") as f:
        test_data = f.readlines()

    train_lines = [" ".join(line.split(",")[1].strip()) for line in train_data]
    train_y = [line.split(",")[0] for line in train_data]
    valid_lines = [" ".join(line.split(",")[1].strip()) for line in valid_data]
    valid_y = [line.split(",")[0] for line in valid_data]
    test_lines = [" ".join(line.split(",")[1].strip()) for line in test_data]
    test_y = [line.split(",")[0] for line in test_data]

    tokenizer.fit_on_texts(train_lines+valid_lines)
    total_words = len(tokenizer.word_index) + 1
    max_seq_len = max([len(x) for x in test_lines+valid_lines+test_lines])
    train_x = tokenizer.texts_to_sequences(train_lines)
    valid_x = tokenizer.texts_to_sequences(valid_lines)
    test_x = tokenizer.texts_to_sequences(test_lines)

    train_x = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=max_seq_len, padding="post", value=0)
    valid_x = keras.preprocessing.sequence.pad_sequences(valid_x, maxlen=max_seq_len, padding="post", value=0)
    test_x = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=max_seq_len, padding="post", value=0)

    return total_words, max_seq_len, tokenizer, [np.array(train_x), np.array(train_y)],\
           [np.array(valid_x), np.array(valid_y)], [np.array(test_x), np.array(test_y)]


vocab_size, max_len, my_tokenizer, my_train_data, my_valid_data, my_test_data = \
    dataset_preparation("train.csv", "valid.csv", "valid.csv")

embed_dim = 300
n_class = 2
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True, input_length=max_len))  # input (batch, input_length)
model.add(keras.layers.LSTM(128, dropout=0.3, return_state=False, return_sequences=False))
model.add(keras.layers.Dense(n_class, activation="softmax"))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.sparse_categorical_crossentropy,
              metrics=["acc"])

callbacks = [keras.callbacks.EarlyStopping(patience=2, monitor='val_acc'),
             keras.callbacks.TensorBoard(log_dir="./logs")]

model.fit(my_train_data[0], my_train_data[1], epochs=10, batch_size=128, callbacks=callbacks,
          validation_data=my_valid_data, shuffle=True)

model.evaluate(*my_test_data, batch_size=32)

export_path = "./model/1"
keras.backend.set_learning_phase(0)

with keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={"input_sentence": model.input},
        outputs={t.name: t for t in model.outputs}
    )


