#data_load/load_dataset.py
from typing import Tuple
import tensorflow_datasets as tfds
import tensorflow as tf

# Type of data returned
Dataset = tf.data.Dataset
DatasetInfo = tfds.core.DatasetInfo

def load_emnist_letters(data_dir: str = 'input_data/datasets') -> Tuple[Tuple[Dataset, Dataset], DatasetInfo]:
    """
    Carga el dataset EMNIST Letters.

    Args:
        data_dir (str): Directorio donde se descargará el dataset.

    Returns:
        Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], tfds.core.DatasetInfo]: 
        Retorna datasets de entrenamiento y prueba, y la info del dataset.
    """
    (ds_train, ds_test), ds_info = tfds.load(
          'emnist/letters:3.1.0',  # especifica la versión exacta que tienes
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
        data_dir='input_data/datasets',
        download=False           # no intenta descargar
    )
    return (ds_train, ds_test), ds_info

