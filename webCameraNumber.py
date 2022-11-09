from keras.models import load_model

model = load_model("mnist_keras_model.h5")

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            g_imag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thr, bin_img = cv2.threshold(g_imag, 110, 255, cv2.THRESH_BINARY_INV)
            contours, hierachy = cv2.findContours(
                bin_img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            try:
                for i in range(len(contours)):
                    contours = contours[i]
                    (x, y), radius = cv2.minEnclosingCircle(contours)
                    if radius > 3:
                        xs, xe = int(x - radius), int(x + radius)
                        ys, ye = int(y - radius), int(y + radius)
                        cv2.rectangle(
                            bin_img,
                            (xs, ys),
                            (xe, ye),
                            (200, 0, 0),
                            2,
                        )
                        roi = bin_img[ys:ye, xs:xe]
                        dst = cv2.resize(
                            roi,
                            dsize=(50, 50),
                            interpolation=cv2.INTER_AREA,
                        )
                        dst = cv2.resize(
                            dst,
                            dsize=(16, 16),
                            interpolation=cv2.INTER_AREA,
                        )
                        A = np.zeros((20, 20))
                        A[2:-2, 2:-2] = dst[:, :]
                        A = A.reshape(-1, 20, 20, 1)
                        num = np.argmax(model.predict(A))
                        cv2.putText(
                            bin_img,
                            str(num),
                            (xs, ys),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1,
                            (200, 0, 0),
                        )

            except Exception as e:
                print(e)
                pass
            cv2.imshow("Image", bin_img)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        else:
            print("No Frame")
            break

else:
    print("Camera not opened")

cap.release()
cv2.destroyAllWindows()
