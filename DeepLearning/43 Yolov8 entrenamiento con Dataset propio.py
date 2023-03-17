# Basado en una guía de roboflow https://blog.roboflow.com/

# para usar yolo en modo CLI se  puede ejecutar desde la consola sin la necesidad de archivo el comando con los
# argumentos para entrenar, validar o ejecutar inferencias en modelos y no necesita realizar ninguna modificación
# en el código
'''
yolo task=detect    mode=train    model=yolov8n.yaml      args...
          classify       predict        yolov8n-cls.yaml  args...
          segment        val            yolov8n-seg.yaml  args...
                         export         yolov8n.pt        format=onnx  args...
'''

# **************
# INFERENCIA
# **************


# ejemplo:
# yolo task=detect mode=predict model='cursos/DeepLearning/models/yolov8n.pt' conf=0.25 source='cursos/DeepLearning/images/bear.jpg' save=True
# lo guarda en /runs/detect/predict

# con Python SDK
from ultralytics import YOLO
model = YOLO('models/yolov8n.pt')
results = model.predict(source='images/goldfish.jpg', conf=0.25)

print(results[0].boxes.xyxy)
'''
tensor([[126.47407, 108.60471, 218.22226, 224.00000],
        [  0.00000,  35.73561, 162.49223, 161.68588],
        [  0.00000,  35.49623, 173.66141, 213.32800]], device='cuda:0')
        '''

print(results[0].boxes.conf) # tensor([0.67925, 0.65461, 0.35953], device='cuda:0')

print(results[0].boxes.cls) # tensor([14., 14., 14.], device='cuda:0')

# ****************
# Entrenamiento
# ****************
