import torch
from generator_model import Generator
import config
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft
from coremltools.models.neural_network import quantization_utils
import coremltools.optimize.coreml._post_training_quantization

def save_model_quantization():
    #квантизация модели при помощи инструментов coremltools
    print("start")
    model_fp32 = ct.models.MLModel('model_comics.mlmodel')
    model_fp16 = quantization_utils.quantize_weights(model_fp32, nbits=16)
    model_fp16.save("model_comics_fp16.mlmodel")
    print("end")
def convert_to_coreml():
    #Подгружаем нормализованную генеративную модель
    model = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    #Подгружаем веса обученной модели
    checkpoint = torch.load(config.CHECKPOINT_GEN_Z, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    #Cоздаем массив нужной размерности для трассировки модели
    example_input = torch.randn((2, 3, 256, 256))
    #Переводим модель в состояние оценки
    model.eval()
    #Создаем трассированную модель
    traced_model = torch.jit.trace(model, example_input)
    #При помощи ImageType нормализуем входные данные для модели
    scale = 1/(0.5*255.0)
    bias = [- 0.5 / 0.5, - 0.5 / 0.5, - 0.5 / 0.5]
    image_input = ct.ImageType(name="input_1",
                               shape=example_input.shape,
                               scale=scale,
                               bias=bias)
    #Конвертируем модель в формат Core ML
    model = ct.convert(
        traced_model,
        inputs=[image_input],
    )
    #Сохраняем сконвертированную модель
    model.save("model_comics.mlmodel")
    #Указываем, что выходной слой выдает изображение размером 256 на 256 пикселей
    spec = ct.utils.load_spec("model_comics.mlmodel")
    output = spec.description.output[0]
    output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    output.type.imageType.height = 256
    output.type.imageType.width = 256

    ct.utils.save_spec(spec, "model_comics.mlmodel")
    print(spec.description)

convert_to_coreml()
save_model_quantization()
