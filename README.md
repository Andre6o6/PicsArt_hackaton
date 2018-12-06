# PicsArt hackaton
Это код моего submission на конкурс [PicsArt hackaton](https://picsart.ai/ru/contest).

## О конкурсе
Стояла задачу семантической сегментации изображений: отделить пользователей от фона на их фотографиях.
В обучающей выборке было около 1500 фотографий, в тестовой - 2500. В качестве метрики использовался коэффициент Дайса.

![dice](https://wikimedia.org/api/rest_v1/media/math/render/svg/a80a97215e1afc0b222e604af1b2099dc9363d3b)

## Мое решение
Для решения задачи я использовал LinkNet-34:

![linknet image](https://www.researchgate.net/publication/323570662/figure/fig2/AS:601018588991488@1520305406425/Fig-These-segmentation-networks-are-based-on-encoder-decoder-network-of-U-Net-family.png)

Это нейронная сеть вида энкодер-декодер с т.н. skip-connections. В качестве энкодера я использовал предобученную сеть ResNet34. Для улучшения точности я использовал аугментацию изображений - random-crop, отражение по горизонтали, перевод в оттенки серого.

Вещи которые я пробовал, но они не получились:
- более сложная сеть (ResNet50)
- ансамбли
- TTA

