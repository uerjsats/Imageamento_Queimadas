import cv2
import numpy as np
import imageio

def detect_fire(frame):
    """
    Detecta áreas de calor (possível fogo) na imagem, diferenciando entre focos ativos e áreas de incêndio maiores.
    """
    # Converte o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define dois limites de intensidade: um para focos (verdes) e outro para incêndios maiores (vermelhos)
    _, thresholded_focos = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    _, thresholded_incendios = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Aplica operações morfológicas para melhorar as máscaras
    kernel_focos = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_incendios = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    thresholded_focos = cv2.morphologyEx(thresholded_focos, cv2.MORPH_CLOSE, kernel_focos)
    thresholded_focos = cv2.morphologyEx(thresholded_focos, cv2.MORPH_OPEN, kernel_focos)

    thresholded_incendios = cv2.morphologyEx(thresholded_incendios, cv2.MORPH_CLOSE, kernel_incendios)
    thresholded_incendios = cv2.morphologyEx(thresholded_incendios, cv2.MORPH_OPEN, kernel_incendios)

    # Encontra contornos nas áreas detectadas
    contours_focos, _ = cv2.findContours(thresholded_focos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_incendios, _ = cv2.findContours(thresholded_incendios, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_count = 0  # Contador para os focos de incêndio
    major_fire_count = 0  # Contador para os incêndios maiores

    for contour in contours_focos:
        if cv2.contourArea(contour) > 20:  # Detecta focos pequenos (verdes)
            if fire_count < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                fire_count += 1

    for contour in contours_incendios:
        if cv2.contourArea(contour) > 50:  # Detecta incêndios maiores (vermelhos)
            if major_fire_count < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                major_fire_count += 1

    # Retorna o frame com os quadrados desenhados e contagens
    return frame, (fire_count > 0 or major_fire_count > 0), fire_count, major_fire_count

print("Iniciando análise de GIF para detecção de queimadas...")

# Carrega o GIF
gif_path = "c:/Users/Helio/Downloads/Monitoramento de Queimadas-20241213T234000Z-001/Monitoramento de Queimadas/JvQuAigCEiZFHpjvS9uefS-1200-80.gif"
resize_factor = 2.0  # Fator de redimensionamento (200% do tamanho original)
try:
    reader = imageio.get_reader(gif_path, mode='I')  # Abre o GIF em modo iterativo
    for idx, frame in enumerate(reader):
        # Converte para BGR para uso no OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Aumenta a resolução do frame
        height, width = frame.shape[:2]
        new_dim = (int(width * resize_factor), int(height * resize_factor))
        frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_CUBIC)

        # Processa cada frame
        processed_frame, fire_detected, fire_count, major_fire_count = detect_fire(frame)

        # Exibe o quadro processado
        cv2.imshow("Detecção de Queimadas - GIF", processed_frame)

        # Exibe alerta no console, se necessário
        if fire_detected:
            print(f"ALERTA: Focos de incêndio detectados no frame {idx}! ({fire_count} focos verdes, {major_fire_count} focos vermelhos)")

        # Aguarda um curto intervalo para simular a reprodução do GIF
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    reader.close()
except Exception as e:
    print(f"Erro ao carregar o GIF: {e}")
    exit()

cv2.destroyAllWindows()
print("Análise do GIF finalizada.")
