from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

train_data = object_detector.DataLoader.from_pascal_voc(
    'train',
    'train',
    ['hama_burung'],
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'valid',
    'valid',
    ['hama_burung'],
)
spec = object_detector.EfficientDetSpec(model_name='efficientdet-lite0', uri="https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1")
model = object_detector.create(train_data, model_spec=spec, batch_size=4, train_whole_model=True, validation_data=val_data)
model.export(export_dir='.', tflite_filename='birds_model.tflite')