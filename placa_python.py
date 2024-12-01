import cv2
import pytesseract
import numpy as np
import os

# Definindo o caminho para o executável do Tesseract
# Verifique se o Tesseract está instalado no local correto
tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Altere esse caminho se necessário

# Verifica se o caminho para o Tesseract está correto
if not os.path.isfile(tesseract_path):
    raise FileNotFoundError(f"O Tesseract não foi encontrado em {tesseract_path}. Verifique a instalação.")

# Configura o caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def reconhecer_placa(imagem_path):
    # Carregar a imagem
    imagem = cv2.imread(imagem_path)
    
    # Verificar se a imagem foi carregada corretamente
    if imagem is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return None

    # Converter para escala de cinza
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar um filtro de desfoque para reduzir ruídos
    imagem_blur = cv2.GaussianBlur(imagem_gray, (5, 5), 0)

    # Detecção de bordas usando Canny
    bordas = cv2.Canny(imagem_blur, 100, 200)

    # Encontrar contornos na imagem
    contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar os contornos por área
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    for contorno in contornos:
        # Aproximar o contorno para um polígono
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)

        # Verificar se o polígono tem 4 vértices (placa retangular)
        if len(aprox) == 4:
            placa_contorno = aprox
            break

    # Criar uma máscara para extrair a região da placa
    mascara = np.zeros(imagem_gray.shape, dtype="uint8")
    cv2.drawContours(mascara, [placa_contorno], -1, 255, 1)

    # Aplicar a máscara na imagem original
    imagem_placa = cv2.bitwise_and(imagem, imagem, mask=mascara)

    # Cortar a região da placa
    (x, y, w, h) = cv2.boundingRect(placa_contorno)
    placa_imagem = imagem_gray[y:y+h, x:x+w]

    # Usar OCR para reconhecer o texto na região da placa
    texto = pytesseract.image_to_string(placa_imagem, config='--psm 8')  # Configuração para reconhecer texto em uma linha

    return texto.strip()

# Caminho para a imagem
imagem_path = r"C:\placa\carro.jpg"  # Altere o caminho da imagem conforme necessário

# Reconhecer a placa
placa = reconhecer_placa(imagem_path)

if placa:
    print("Placa do carro reconhecida:", placa)
