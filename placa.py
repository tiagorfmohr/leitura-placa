import cv2
import torch
import easyocr
from PIL import Image

def recognize_plate(img_path, model_name='yolov5s', lang='br', confidence_threshold=0.5):
    try:
        # Carregar o modelo YOLOv5
        model = torch.hub.load('ultralytics/yolov5', model_name)

        # Ler a imagem
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro: Imagem não encontrada: {img_path}")
            return

        # Realizar a detecção de objetos na imagem
        results = model(img)

        # Exibir todas as detecções para verificar se há placas detectadas
        results.show()  # Isso exibe as detecções em uma janela pop-up

        # Filtrar as detecções para obter apenas placas com alta confiança
        plates = []
        for *xyxy, conf, cls in results.xywh[0]:
            if conf >= confidence_threshold:
                # Coletar a caixa delimitadora (bounding box) da placa
                x1, y1, x2, y2 = map(int, xyxy)
                plate_img = img[y1:y2, x1:x2]
                plates.append(plate_img)

        # Se não encontrar placas, sair
        if not plates:
            print("Nenhuma placa detectada.")
            return

        # Reconhecimento OCR
        reader = easyocr.Reader([lang])
        for plate_img in plates:
            result = reader.readtext(plate_img)

            # Exibir resultados detalhados
            print("Resultados do OCR:")
            for bbox, text, prob in result:
                print(f"Texto: {text}, Probabilidade: {prob}")

        # Salvar imagem com resultados (se desejar)
        cv2.imwrite("resultado.jpg", img)

        cv2.imshow('Resultado', img)
        cv2.waitKey(0)

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Exemplo de uso
recognize_plate('carro.jpg', model_name='yolov5m', lang='pt-br', confidence_threshold=0.5)
