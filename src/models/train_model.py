import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tensorflow.keras import layers, models, applications

def main():
    # ---------------------------------------------------------
    # 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS
    # ---------------------------------------------------------
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent.parent / "data" / "clean"
    
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20

    print(f"Buscando imágenes en: {data_dir}")

    # ---------------------------------------------------------
    # 2. PREPARACIÓN DE LOS CONJUNTOS DE DATOS (DATA PIPELINE)
    # ---------------------------------------------------------
    print("\nCargando datasets...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    val_test_dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Dividimos el 20% de validación en 10% validación real y 10% prueba
    val_batches = tf.data.experimental.cardinality(val_test_dataset)
    val_dataset = val_test_dataset.take(val_batches // 2)
    test_dataset = val_test_dataset.skip(val_batches // 2)

    # ---------------------------------------------------------
    # 3. CÁLCULO DE PESOS DE CLASE (PARA EL DESBALANCE)
    # ---------------------------------------------------------
    class_names = train_dataset.class_names
    print(f"Clases detectadas: {class_names}")

    y_train = np.concatenate([y for x, y in train_dataset], axis=0)
    y_train_indices = np.argmax(y_train, axis=1)

    pesos = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_indices),
        y=y_train_indices
    )
    class_weights_dict = dict(enumerate(pesos))
    print(f"Pesos aplicados para balancear: {class_weights_dict}")

    # Optimización de carga en memoria
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # ---------------------------------------------------------
    # 4. ARQUITECTURA DEL MODELO (MOBILENETV3)
    # ---------------------------------------------------------
    print("\nConstruyendo la arquitectura del modelo...")
    
    # Capas de Data Augmentation al vuelo
    data_augmentation = tf.keras.Sequential([
      layers.RandomFlip("horizontal_and_vertical"),
      layers.RandomRotation(0.2),
      layers.RandomZoom(0.2),
      layers.RandomContrast(0.1),
    ], name="data_augmentation_layer")

    # Base convolucional preentrenada
    base_model = applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=True 
    )
    base_model.trainable = False 

    # Ensamblaje
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(9, activation='softmax', name='prediccion_enfermedad')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # ---------------------------------------------------------
    # 5. ENTRENAMIENTO (MODEL.FIT)
    # ---------------------------------------------------------
    print("\n¡Iniciando el entrenamiento!")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        class_weight=class_weights_dict
    )

    # ---------------------------------------------------------
    # 6. GUARDAR EL MODELO FINAL
    # ---------------------------------------------------------
    model_save_path = script_dir / "mobilenetv3_maiz_v1.keras"
    model.save(model_save_path)
    print(f"\n✅ Entrenamiento finalizado. Modelo guardado en: {model_save_path}")

if __name__ == "__main__":
    main()