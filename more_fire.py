import cv2
import numpy as np
import imageio

def detect_fire(frame):
    """
    Detecta áreas de calor (possível fogo) na imagem.
    """
    # Converte o quadro para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os limites de cor para detecção de fogo
    lower_bound = np.array([0, 50, 200])  # Tons avermelhados/amarelados
    upper_bound = np.array([35, 255, 255])

    # Cria uma máscara para a cor do fogo
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Encontra contornos nas áreas detectadas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Reduz a área mínima para capturar mais focos em imagens de baixa resolução
        if cv2.contourArea(contour) > 100:  # Ajuste de 500 para 100
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Corrigido para desenhar no frame original

    return frame, len(contours) > 0

print("Iniciando análise de GIF para detecção de queimadas...")

# Carrega o GIF
gif_path = "c:/Users/Helio/Downloads/Monitoramento de Queimadas-20241213T234000Z-001/Monitoramento de Queimadas/downtoearth_2024-08_11f34c96-2677-40ae-a212-5ece25af0ba7_1000093037.gif"
try:
    reader = imageio.get_reader(gif_path, mode='I')  # Abre o GIF em modo iterativo
    for idx, frame in enumerate(reader):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Converte para BGR para uso no OpenCV

        # Processa cada frame
        processed_frame, fire_detected = detect_fire(frame)

        # Exibe o quadro processado
        cv2.imshow("Detecção de Queimadas - GIF", processed_frame)

        # Exibe alerta no console, se necessário
        if fire_detected:
            print(f"ALERTA: Possível foco de queimada detectado no frame {idx}!")

        # Aguarda um curto intervalo para simular a reprodução do GIF
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    reader.close()
except Exception as e:
    print(f"Erro ao carregar o GIF: {e}")
    exit()

cv2.destroyAllWindows()
print("Análise do GIF finalizada.")
