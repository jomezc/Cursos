# Creando un adaptador de Dataset
# Based on to https://medium.com/data-science-at-microsoft/training-efficientdet-on-custom-data-with-pytorch-lightning-using-an-efficientnetv2-backbone-1cdf3bd7921f
from pathlib import Path

from pathlib import Path

dataset_path = Path('/home/jesus/OneDrive/Jgomcano/Datashets/CarObjectDetection')
print(list(dataset_path.iterdir()))
'''
[PosixPath('/home/jesus/OneDrive/Jgomcano/Datashets/CarObjectDetection/sample_submission.csv'), 
PosixPath('/home/jesus/OneDrive/Jgomcano/Datashets/CarObjectDetection/testing_images'), 
PosixPath('/home/jesus/OneDrive/Jgomcano/Datashets/CarObjectDetection/train_solution_bounding_boxes (1).csv'), 
PosixPath('/home/jesus/OneDrive/Jgomcano/Datashets/CarObjectDetection/training_images')]
'''

# directorio con los datos de las imágenes
train_data_path = dataset_path / 'training_images'

'''
Las anotaciones para este conjunto de datos están en forma de archivo csv, que asocia el nombre de la imagen con las 
anotaciones correspondientes, podemos ver el formato de esto cargándolo en un marco de datos.
'''
import pandas as pd

df = pd.read_csv(dataset_path / 'train_solution_bounding_boxes (1).csv')
print(df.head())
'''
             image        xmin        ymin        xmax        ymax
0   vid_4_1000.jpg  281.259045  187.035071  327.727931  223.225547
1  vid_4_10000.jpg   15.163531  187.035071  120.329957  236.430180
2  vid_4_10040.jpg  239.192475  176.764801  361.968162  236.430180
3  vid_4_10020.jpg  496.483358  172.363256  630.020260  231.539575
4  vid_4_10060.jpg   16.630970  186.546010  132.558611  238.386422
'''

## Adaptador de conjuntos de datos
'''

Para poder utilizarlo en nuestro modelo, primero debemos crear un Adaptador de conjuntos de datos, que convertirá el 
formato del conjunto de datos en bruto en una imagen y las anotaciones correspondientes para introducirlas en el modelo.

En primer lugar, definiremos algunas funciones útiles para poder trazar imágenes con sus correspondientes cuadros 
delimitadores'''

import matplotlib.pyplot as plt
from matplotlib import patches


def get_rectangle_edges_from_pascal_bbox(bbox):
    xmin_top_left, ymin_top_left, xmax_bottom_right, ymax_bottom_right = bbox

    bottom_left = (xmin_top_left, ymax_bottom_right)
    width = xmax_bottom_right - xmin_top_left
    height = ymin_top_left - ymax_bottom_right

    return bottom_left, width, height


def draw_pascal_voc_bboxes(
        plot_ax,
        bboxes,
        get_rectangle_corners_fn=get_rectangle_edges_from_pascal_bbox,
):
    for bbox in bboxes:
        bottom_left, width, height = get_rectangle_corners_fn(bbox)

        rect_1 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=4,
            edgecolor="black",
            fill=False,
        )
        rect_2 = patches.Rectangle(
            bottom_left,
            width,
            height,
            linewidth=2,
            edgecolor="white",
            fill=False,
        )

        # Add the patch to the Axes
        plot_ax.add_patch(rect_1)
        plot_ax.add_patch(rect_2)


def show_image(
        image, bboxes=None, draw_bboxes_fn=draw_pascal_voc_bboxes, figsize=(10, 10)
):
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    if bboxes is not None:
        draw_bboxes_fn(ax, bboxes)

    plt.show()


'''
Normalmente, en este punto, crearíamos un conjunto de datos PyTorch para introducir estos datos en el bucle de 
entrenamiento. Sin embargo, parte de este código, como la normalización de la imagen y la transformación de las 
etiquetas al formato requerido, no son específicos de este problema y deberán aplicarse independientemente del conjunto 
de datos que se utilice. 
Por lo tanto, vamos a centrarnos por ahora en la creación de una clase CarsDatasetAdaptor, que convertirá el formato 
específico del conjunto de datos en bruto en una imagen y las anotaciones correspondientes.  A continuación se presenta 
una implementación de esta clase'''
from pathlib import Path
import PIL
import numpy as np
import timm


class CarsDatasetAdaptor:
    # Cuando se usa un conjunto de datos diferente, ¡esta es la parte que cambia!

    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()

    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        #  Para EfficientDet, las clases deben comenzar en 1, y -1 se usa para la clase de "fondo".

        image_name = self.images[index]
        # Imagen en formato PIL
        image = PIL.Image.open(self.images_dir_path / image_name)

        # matriz numpy de forma [N, 4] que contiene los cuadros delimitadores de la verdad básica en formato Pascal VOC
        pascal_bboxes = self.annotations_df[self.annotations_df.image == image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values

        # una matriz numpy de forma N que contiene las etiquetas de clase verdaderas
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index

    # Mostrar imagen
    def show_image(self, index):
        # image_id: un identificador único que puede utilizarse para identificar la imagen y el método __len__.
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)


'''Como podemos ver, en este caso, esta clase simplemente envuelve el marco de datos proporcionado con el conjunto de 
datos. Ahora podemos crear una instancia de esta clase para proporcionar una interfaz limpia para ver los datos de 
entrenamiento. Como este conjunto de datos sólo contiene una única clase, en este caso siempre se devuelven unos. 
Además, como image_id puede ser cualquier identificador único asociado a la imagen, aquí sólo hemos utilizado el 
índice de la imagen en el conjunto de datos.'''

cars_train_ds = CarsDatasetAdaptor(train_data_path, df)
cars_train_ds.show_image(0)
cars_train_ds.show_image(3)

### CREANDO EL MODELO

# Revisar los modelos disponibles en eff det ( los tf_ inicialmente entrenados para tensorflow
#  para usar estos pesos en PyTorch, se han realizado ciertas modificaciones (como implementar el mismo relleno), lo
#  que significa que estos modelos pueden ser más lentos durante el entrenamiento y la inferencia.

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

print(f' numero de ocnfiguraciones: {len(efficientdet_model_param_dict)}')
print(list(efficientdet_model_param_dict.keys())[::3])
'''
 numero de ocnfiguraciones: 47
['efficientdet_d0', 'efficientdet_d3', 'efficientdetv2_dt', 'cspresdet50', 'cspdarkdet53', 'mixdet_l', 
'mobiledetv3_large', 'efficientdet_q2', 'efficientdet_em', 'tf_efficientdet_d1', 'tf_efficientdet_d4', 
'tf_efficientdet_d7', 'tf_efficientdet_d1_ap', 'tf_efficientdet_d4_ap', 'tf_efficientdet_lite1', 'tf_efficientdet_lite3x']
'''

# también podemos usar cualquier modelo de timm como nuestra columna vertebral de EfficientDet.
print(timm.list_models('tf_efficientnetv2_*'))
'''['tf_efficientnetv2_b0', 'tf_efficientnetv2_b1', 'tf_efficientnetv2_b2', 'tf_efficientnetv2_b3', 
'tf_efficientnetv2_l', 'tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21k', 'tf_efficientnetv2_m', 
'tf_efficientnetv2_m_in21ft1k', 'tf_efficientnetv2_m_in21k', 'tf_efficientnetv2_s', 'tf_efficientnetv2_s_in21ft1k', 
'tf_efficientnetv2_s_in21k', 'tf_efficientnetv2_xl_in21ft1k', 'tf_efficientnetv2_xl_in21k']
'''

# Para usar uno de estos modelos, primero debemos registrarlo como una configuración de EfficientDet agregando un
# diccionario a "ficientdet_model_param_dict". Vamos a crear una función que haga esto por nosotros y luego cree el
# modelo EfficientDet usando la maquinaria de effdet

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

'''
Aquí, una vez creado el modelo EfficientDet, modificamos el encabezado de clasificación en función del número de clases 
para nuestra tarea actual. Hemos establecido el tamaño de imagen predeterminado en 512, como se usa en el documento. 
Debido a la arquitectura de EfficientDet, el tamaño de la imagen de entrada debe ser divisible por 128 . 
https://medium.com/@nainaakash012/efficientdet-scalable-and-efficient-object-detection-ea05ccd28427
Aquí, usamos el
tamaño predeterminado de 512. Ahora podemos usarlo para crear nuestro modelo PyTorch EfficientDet.'''


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(
        name='tf_efficientnetv2_l',
        backbone_name='tf_efficientnetv2_l',
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url='', )

    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


# Creación de un conjunto de datos y un módulo de datos EfficientDet
''' Comencemos definiendo algunas transformaciones, que deben aplicarse antes de pasar las imágenes y las etiquetas al 
modelo. Para ello, podemos utilizar la excelente biblioteca Albumentations , que contiene una amplia variedad de métodos
 de aumento de datos.

 Aquí, con el objetivo de simplificar las cosas, mantenemos solo el preprocesamiento esencial durante la validación: 
 como la columna vertebral se entrenó previamente, necesitamos normalizar la imagen utilizando la media y la desviación 
 estándar del conjunto de datos de ImageNet, así como cambiar el tamaño de la imagen. y conviértalo en un tensor, y 
 agregue un giro horizontal mientras entrena. ¡Podemos ver que tenemos que pasar los cuadros delimitadores a estas 
 transformaciones, porque Albumentations también aplica cualquier transformación a las etiquetas!
 '''
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


'''
Ahora que hemos definido nuestras transformaciones, podemos pasar a definir un conjunto de datos para envolver nuestro 
adaptador de conjunto de datos y aplicar las transformaciones. El único problema a tener en cuenta es que EfficientDet 
requiere los cuadros delimitadores en formato YXYX . Podemos ver la implementación de esto a continuación:
'''

from torch.utils.data import Dataset
import torch


class EfficientDetDataset(Dataset):
    def __init__(
            self, dataset_adaptor, transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }

        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][
                                            :, [1, 0, 3, 2]
                                            ]  # convert to yxyx

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


'''
Si bien ahora podemos usar este conjunto de datos para crear un PyTorch DataLoader estándar, PyTorch-lightning 
proporciona una clase DataModule, que podemos usar para agrupar todos nuestros componentes relacionados con los datos. 
La interfaz es bastante intuitiva, y podemos implementarla de la siguiente manera:
'''

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class EfficientDetDataModule(LightningDataModule):

    def __init__(self,
                 train_dataset_adaptor,
                 validation_dataset_adaptor,
                 train_transforms=get_train_transforms(target_img_size=512),
                 valid_transforms=get_valid_transforms(target_img_size=512),
                 num_workers=4,
                 batch_size=8):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds, transforms=self.train_tfms
        )

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds, transforms=self.valid_tfms
        )

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return valid_loader

    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids


# Definición del ciclo de entrenamiento
'''En PyTorch-lightning, vinculamos el modelo, el ciclo de entrenamiento y el optimizador en un LightningModule. 
Entonces, en lugar de tener que definir nuestros propios bucles para iterar sobre cada cargador de datos, podemos hacer 
lo siguiente'''

from numbers import Number
from typing import List
from functools import singledispatch

import numpy as np
import torch

from fastcore.dispatch import typedispatch
from pytorch_lightning import LightningModule
# from pytorch_lightning.core.decorators import auto_move_data

from ensemble_boxes import ensemble_boxes_wbf


def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):
    def __init__(
            self,
            num_classes=1,
            img_size=512,
            prediction_confidence_threshold=0.2,
            learning_rate=0.0002,
            wbf_iou_threshold=0.44,
            inference_transforms=get_valid_transforms(target_img_size=512),
            model_architecture='tf_efficientnetv2_l',
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms

    # @auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(
            "train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
            logger=True
        )
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences

        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (
                images_tensor.shape[-1] != self.img_size
                or images_tensor.shape[-2] != self.img_size
        ):
            raise ValueError(
                f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})"
            )

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )

        detections = self.model(images_tensor.to(self.device), dummy_targets)[
            "detections"
        ]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences

    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                            np.array(bboxes)
                            * [
                                im_w / self.img_size,
                                im_h / self.img_size,
                                im_w / self.img_size,
                                im_h / self.img_size,
                            ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes


dm = EfficientDetDataModule(train_dataset_adaptor=cars_train_ds,
                            validation_dataset_adaptor=cars_train_ds,
                            num_workers=4,
                            batch_size=2)

model = EfficientDetModel(
    num_classes=1,
    img_size=512
)

# Entrenando al modelo


# ############################################
# # Optimizar configuración operaciones cuda #
# ############################################
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from pytorch_lightning import Trainer

trainer = Trainer(accelerator='gpu', devices=1, max_epochs=5, num_sanity_val_steps=1)
trainer.fit(model, dm)

'''File "/home/jesus/OneDrive/Jgomcano/IA/venv/curso_DL_3_10/lib/python3.10/site-packages/effdet/anchors.py", line 398, in batch_label_anchors
    box_targets[count:count + steps].view([feat_size[0], feat_size[1], -1]))
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
https://stackoverflow.com/questions/66750391/runtimeerror-view-size-is-not-compatible-with-input-tensors-size-and-stride-a
Just replace the view() function with reshape() function as suggested in the error and it works.
'''

# Salvar el modelo
torch.save(model.state_dict(), 'trained_effdet')

# cargar el modelo
model = EfficientDetModel(
    num_classes=1,
    img_size=512
)

model.load_state_dict(torch.load('trained_effdet'))

# Usa el modelo para inferencia
model.eval()
print(model)

# Ahora podemos utilizar nuestro adaptador de conjunto de datos para cargar una selección de imágenes
image1, truth_bboxes1, _, _ = cars_train_ds.get_image_and_labels_by_idx(0)
image2, truth_bboxes2, _, _ = cars_train_ds.get_image_and_labels_by_idx(1)
images = [image1, image2]

# y la función de predicción del modelo para obtener los cuadros delimitadores predichos para estas imágenes
predicted_bboxes, predicted_class_confidences, predicted_class_labels = model.predict(images)


# Podemos visualizar estas predicciones utilizando una función
def compare_bboxes_for_image(
        image,
        predicted_bboxes,
        actual_bboxes,
        draw_bboxes_fn=draw_pascal_voc_bboxes,
        figsize=(20, 20),
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.imshow(image)
    ax1.set_title("Prediction")
    ax2.imshow(image)
    ax2.set_title("Actual")

    draw_bboxes_fn(ax1, predicted_bboxes)
    draw_bboxes_fn(ax2, actual_bboxes)

    plt.show()


compare_bboxes_for_image(image1, predicted_bboxes=predicted_bboxes[0], actual_bboxes=truth_bboxes1.tolist())

compare_bboxes_for_image(image2, predicted_bboxes=predicted_bboxes[1], actual_bboxes=truth_bboxes2.tolist())

'''
Uso de ganchos de modelo para depurar manualmente

Una característica de PyTorch Lightning es que utiliza métodos o "ganchos" para representar cada parte del proceso de 
entrenamiento. Aunque perdemos algo de visibilidad sobre nuestro bucle de entrenamiento cuando utilizamos el Trainer, 
podemos utilizar estos ganchos para depurar fácilmente cada paso.

Por ejemplo, podemos utilizar un gancho definido en nuestro DataModule para obtener el dataloader que se utilizará 
durante la validación y utilizar esto para agarrar un lote.
'''
loader = dm.val_dataloader()
dl_iter = iter(loader)
batch = next(dl_iter)

'''
Podemos utilizar este lote para ver exactamente lo que el modelo calculó durante la validación. Como lightning se 
encarga de mover los datos al dispositivo correcto durante el entrenamiento, por simplicidad, lo haremos en la cpu para 
no tener que mover manualmente todos los tensores de cada lote al dispositivo.'''
device = model.device;
device
device(type='cpu')
output = model.validation_step(batch=batch, batch_idx=0)
print(output)

'''
Aquí podemos ver que se devuelve la pérdida para el lote, así como las predicciones y los objetivos.

Para calcular las métricas, para la época, necesitamos obtener las predicciones correspondientes a cada lote. Como el 
método `validation_step` será llamado para cada lote, vamos a definir una función para agregar las salidas. 

Aquí, por simplicidad, parchearemos esta función a la clase EfficientDet usando un decorador de conveniencia de fastcore
 - pagamos un enorme precio de rendimiento por ser Python un lenguaje dinámico, ¡más vale que lo aprovechemos!'''

from fastcore.basics import patch


@patch
def aggregate_prediction_outputs(self: EfficientDetModel, outputs):
    detections = torch.cat(
        [output["batch_predictions"]["predictions"] for output in outputs]
    )

    image_ids = []
    targets = []
    for output in outputs:
        batch_predictions = output["batch_predictions"]
        image_ids.extend(batch_predictions["image_ids"])
        targets.extend(batch_predictions["targets"])

    (
        predicted_bboxes,
        predicted_class_confidences,
        predicted_class_labels,
    ) = self.post_process_detections(detections)

    return (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    )


'''En la documentación de PyTorch-lightning 
(ver https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-epoch-level-metrics), 
podemos ver que podemos añadir un hook adicional `validation_epoch_end` que es llamado después de que todos los lotes 
hayan sido procesados; al final de cada epoch, una lista de los resultados de los pasos son pasados a este hook.

Utilicemos este hook para calcular la pérdida global de validación, así como las métricas COCO utilizando el paquete `
objdetecteval`. podemos utilizar la salida que acabamos de calcular al evaluar un único lote de validación, pero este 
enfoque también se extendería a la evaluación del bucle de validación durante el entrenamiento con lightning.
!pip install git+https://github.com/alexhock/object-detection-metrics'''

from objdetecteval.metrics.coco_metrics import get_coco_stats


@patch
def validation_epoch_end(self: EfficientDetModel, outputs):
    """Compute and log training loss and accuracy at the epoch level."""

    validation_loss_mean = torch.stack(
        [output["loss"] for output in outputs]
    ).mean()

    (
        predicted_class_labels,
        image_ids,
        predicted_bboxes,
        predicted_class_confidences,
        targets,
    ) = self.aggregate_prediction_outputs(outputs)

    truth_image_ids = [target["image_id"].detach().item() for target in targets]
    truth_boxes = [
        target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
    ]  # convert to xyxy for evaluation
    truth_labels = [target["labels"].detach().tolist() for target in targets]

    stats = get_coco_stats(
        prediction_image_ids=image_ids,
        predicted_class_confidences=predicted_class_confidences,
        predicted_bboxes=predicted_bboxes,
        predicted_class_labels=predicted_class_labels,
        target_image_ids=truth_image_ids,
        target_bboxes=truth_boxes,
        target_class_labels=truth_labels,
    )['All']

    return {"val_loss": validation_loss_mean, "metrics": stats}


model.validation_epoch_end([output])

'''
Para la inferencia

También podemos utilizar la función predecir directamente sobre las imágenes procesadas devueltas por nuestro cargador 
de datos. Vamos a descomprimir el lote para obtener sólo las imágenes, ya que no necesitamos las etiquetas para la 
inferencia.

Gracias al decorador typedispatch, podemos utilizar la misma firma de función predict en estos tensores.

Es importante señalar en este punto que las imágenes proporcionadas por el cargador de datos ya han sido transformadas y
 escaladas a un tamaño de 512. Por lo tanto, los cuadros delimitadores predichos por el cargador de datos no son los 
 mismos. Por lo tanto, los cuadros delimitadores predichos serán relativos para una imagen de 512. Por lo tanto, para 
 visualizar estas predicciones en la imagen original, debemos reescalarla.'''
images, annotations, targets, image_ids = batch
predicted_bboxes, predicted_class_labels, predicted_class_confidences = model.predict(images)
image, _, _, _ = cars_train_ds.get_image_and_labels_by_idx(0)
show_image(image.resize((512, 512)), predicted_bboxes[0])