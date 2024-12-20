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

    # Realça as áreas detectadas na imagem original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Encontra contornos nas áreas detectadas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignora pequenos contornos para evitar falsos positivos
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, len(contours) > 0

print("Iniciando análise de GIF para detecção de queimadas...")

# Carrega o GIF
gif_path = "c:/Users/Helio/Downloads/Monitoramento de Queimadas-20241213T234000Z-001/Monitoramento de Queimadas/banner_jenny.gif"
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

    # Realça as áreas detectadas na imagem original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Encontra contornos nas áreas detectadas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignora pequenos contornos para evitar falsos positivos
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, len(contours) > 0

print("Iniciando análise de GIF para detecção de queimadas...")

# Carrega o GIF
gif_path = "c:/Users/Helio/Downloads/Monitoramento de Queimadas-20241213T234000Z-001/Monitoramento de Queimadas/banner_jenny.gif"
try:
    gif = imageio.mimread(gif_path)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif]  # Converte para BGR para uso no OpenCV
except Exception as e:
    print(f"Erro ao carregar o GIF: {e}")
    exit()

for idx, frame in enumerate(frames):
    # Processa cada frame do GIF
    processed_frame, fire_detected = detect_fire(frame)

    # Exibe o quadro processado
    cv2.imshow("Detecção de Queimadas - GIF", processed_frame)

    # Exibe alerta no console, se necessário
    if fire_detected:
        print(f"ALERTA: Possível foco de queimada detectado no frame {idx}!")

    # Aguarda um curto intervalo para simular a reprodução do GIF
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Análise do GIF finalizada.")

try:
    gif = imageio.mimread(gif_path)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in gif]  # Converte para BGR para uso no OpenCV
except Exception as e:
    print(f"Erro ao carregar o GIF: {e}")
    exit()

for idx, frame in enumerate(frames):
    # Processa cada frame do GIF
    processed_frame, fire_detected = detect_fire(frame)

    # Exibe o quadro processado
    cv2.imshow("Detecção de Queimadas - GIF", processed_frame)

    # Exibe alerta no console, se necessário
    if fire_detected:
        print(f"ALERTA: Possível foco de queimada detectado no frame {idx}!")

    # Aguarda um curto intervalo para simular a reprodução do GIF
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Análise do GIF finalizada.")
