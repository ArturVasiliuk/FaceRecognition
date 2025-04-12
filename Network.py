import shutil
import os
import keras
import tensorflow


data_dir = "./dataset"
train_dir = "train"
val_dir = "val"
test_dir = "test"
test_data_portion = 0.1
val_data_portion = 0.1
nb_images = 9000
img_width, img_height = 225, 225
input_shape = (img_width, img_height, 3)
epochs = 5
batch_size = 32
nb_train_samples = 14400
nb_validation_samples = 1800
nb_test_samples = 1800


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "artur"))
    os.makedirs(os.path.join(dir_name, "unknown"))


def copy_images(start_index, end_index, source_dir, dest_dir):
    for i in range(start_index, end_index):
        shutil.copy2(
            os.path.join(source_dir, "artur." + str(i) + ".png"),
            os.path.join(dest_dir, "artur"),
        )
        shutil.copy2(
            os.path.join(source_dir, "unknown." + str(i) + ".png"),
            os.path.join(dest_dir, "unknown"),
        )


create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)


start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))


copy_images(0, start_val_data_idx, data_dir, train_dir)
copy_images(start_val_data_idx, start_test_data_idx, data_dir, val_dir)
copy_images(start_test_data_idx, nb_images, data_dir, test_dir)


model = keras.Sequential()
model.add(keras.layers.Conv2D(8, (3, 3), input_shape=input_shape))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(16, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(16, (3, 3)))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Activation("sigmoid"))


model.compile(loss="BinaryCrossentropy", optimizer="adam", metrics=["accuracy"])


datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
)
val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
)
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="binary",
)


model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
)


scores = model.predict(test_generator, nb_test_samples // batch_size)
print(f"Точность на тестовых данных: {(scores[1] * 100)}")


model.save("model.keras")
