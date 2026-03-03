from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    sucesso, frame = cap.read()
    if sucesso:
        # Rodando a detecção
        results = model(frame, conf=0.5, verbose=False) # verbose=False limpa o terminal

        # --- NOVA LÓGICA ---
        # 4. Pegamos todos os objetos detectados no frame atual
        for r in results:
            for box in r.boxes:
                # Pegamos o ID da classe (ex: 0 é pessoa, 67 é celular)
                cls = int(box.cls[0])
                nome_objeto = model.names[cls]

                if nome_objeto == "cell phone":
                    print("⚠️ ALERTA: Celular detectado na mão do usuário!")
        # -------------------

        annotated_frame = results[0].plot()
        cv2.imshow("IA com Logica de Alerta", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()