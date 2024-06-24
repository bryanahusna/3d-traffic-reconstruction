import cv2
from m3d_rpn.m3d_rpn import M3DRPN

m3d_rpn = M3DRPN()

im = cv2.imread('images/input2.png')
results = m3d_rpn.predict(im)
for result in results:
    cv2.putText(im, f'{result[-2]}', (int((result[2] + result[4])/2), int((result[3] + result[5])/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imwrite('m3d_rpn_trial.png', im)

print(results)
