import cv2
import numpy as np
import imageio

def detect_fire(frame):
    """
    Detecta áreas de calor (possível fogo) na imagem, baseando-se nas áreas de alta intensidade (brancas).
    """
    # Converte o quadro para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define um limite de intensidade para identificar as áreas mais brilhantes (possíveis focos de incêndio)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Aplica operações morfológicas para melhorar a máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    # Encontra contornos nas áreas detectadas
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_count = 0  # Contador para os focos de incêndio

    for contour in contours:
        # Reduz a área mínima para capturar mais focos em imagens de baixa resolução
        if cv2.contourArea(contour) > 100:  # Ajuste de 500 para 100
            if fire_count < 70:  # Limita a detecção a 70 focos
                # Calcula o retângulo delimitador dos contornos
                x, y, w, h = cv2.boundingRect(contour)
                # Ajusta os quadrados para mostrar exatamente os focos detectados
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                fire_count += 1

    # Retorna o frame com os quadrados desenhados e se algum foco foi detectado
    return frame, fire_count > 0, fire_count

print("Iniciando análise de GIF para detecção de queimadas...")

# Carrega o GIF
gif_path = "c:/Users/Helio/Downloads/Monitoramento de Queimadas-20241213T234000Z-001/Monitoramento de Queimadas/20230818091000-20230818105000_n20-n21-snpp_viirs_conus_daynightband_bcfires_nolabels.gif"
try:
    reader = imageio.get_reader(gif_path, mode='I')  # Abre o GIF em modo iterativo
    for idx, frame in enumerate(reader):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Converte para BGR para uso no OpenCV

        # Processa cada frame
        processed_frame, fire_detected, fire_count = detect_fire(frame)

        # Exibe o quadro processado
        cv2.imshow("Detecção de Queimadas - GIF", processed_frame)

        # Exibe alerta no console, se necessário
        if fire_detected:
            print(f"ALERTA: Possível foco de queimada detectado no frame {idx}! ({fire_count} focos)")

        # Aguarda um curto intervalo para simular a reprodução do GIF
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    reader.close()
except Exception as e:
    print(f"Erro ao carregar o GIF: {e}")
    exit()

cv2.destroyAllWindows()
print("Análise do GIF finalizada.")
