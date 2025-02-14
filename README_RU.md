# Team5:ADS - Обнаружение и предотвращение столкновений

[АНГЛИЙСКАЯ ВЕРСИЯ ЗДЕСЬ](README.md)

## Цель проекта

Этот проект создан для того, чтобы дать вам практический опыт в разработке системы обнаружения и предотвращения столкновений для транспортных средств, используя как виртуальные симуляции, так и реальную аппаратную реализацию. Используя платформы, такие как CARLA, Gazebo или AirSim, вы создадите систему, которая не только обнаруживает препятствия на пути транспортного средства, но и выполняет автономные маневры для предотвращения потенциальных столкновений, обеспечивая при этом безопасность пассажиров.

---

## Обзор проекта

Файлы в каждом каталоге выполняют следующие функции:
- **server:** Эти файлы содержат реализацию FLASK-сервера, который обрабатывает вычисления, выполняемые моделью YOLOPv3.
- **colab:** Код, используемый в Google Colaboratory, работает с Google Drive. Он загружает данные из Google Drive и сохраняет визуализацию обнаруженных полос движения и сегментированной дороги с помощью YOLOPv3 обратно в Google Drive.
- **yolopv3:** Исходные файлы из проекта YOLOPv3.

---

## Контейнер Docker
Dockerfile для сервера находится в [server/docker_arm64](server/docker_arm64) или [server/docker_amd64](server/docker_amd64). Если вы не хотите настраивать среду вручную, вы можете просто загрузить и запустить этот Dockerfile. Он включает зависимости Python и файл весов .pth для YOLOPv3.

Как загрузить и запустить его:

```
docker pull yeongyoo/ads_team:server_flask
docker run --privileged -it --network host -e FLASK_APP=server_yolopv3.py --name ads yeongyoo/ads_team:0.2
```

Как использовать контейнер сервера:
```
docker start ads
```
Ссылка на контейнер в DockerHub: [здесь](https://hub.docker.com/repository/docker/yeongyoo/ads_team/general).

## Код сервера
Этот раздел содержит код FLASK-сервера, который выполняет вычисления машинного обучения и отправляет длину автомобиля.

### server/lib
По сути, файлы Python внутри этого каталога аналогичны [данному репозиторию](https://github.com/jiaoZ7688/YOLOPv3/tree/main/lib). Они помогают запускать модель и поддерживать различные функции.

### server/utils
Эти файлы извлекают оптимальные значения угла поворота и скорости на основе полученных данных. Глубокая нейросеть YOLOPv3 используется для сегментации полос движения и обнаружения объектов. Основные этапы:

1. Запуск модели YOLOPv3 с помощью класса ```ModelRun()```, который включает предобработку и постобработку. Выходные данные имеют следующий формат:
```
{
  "identifier": "уникальный_идентификатор",
  "detections": [
    {
      "class": "car",
      "confidence": 0.95,
      "bbox": [xmin, ymin, xmax, ymax],
      "center": [x, y]
    }
  ],
  "da_seg_mask": [[0, 1, 0], [1, 0, 1]],
  "ll_seg_mask": [[1, 0, 1], [0, 1, 0]]
}
```
2. Преобразование ```ll_seg_mask``` и ```da_seg_mask``` в вид "с высоты птичьего полета" (вертикальный обзор), что улучшает понимание траектории движения.
3. Использование метода "скользящего окна" для обнаружения полос движения и анализа их кривизны. Класс ```SlideWindow()``` возвращает x-координату центра дороги, которая служит важным индикатором кривизны полосы.
4. Используя данные из метода "скользящего окна", определяется угол поворота и скорость. Если обнаружены объекты, считающиеся стоп-сигналами или пешеходами, автомобиль должен остановиться.

#### image2mani.py
Класс Image2Mani отвечает за предложение оптимальных значений угла поворота и скорости. Метод ```run()``` принимает изображение в качестве входных данных и выдает значение угла поворота.

#### modelrun.py
Класс ModelRun выполняет запуск модели YOLOPv2, включая предобработку и постобработку. Метод ```run()``` возвращает обнаруженные объекты, сегментированную область движения и сегментированные полосы движения.

#### slidewindow.py
Класс SlideWindow помогает компьютеру обнаружить местоположение полос движения с помощью метода "скользящего окна". Он определяет, где находятся полосы, и предоставляет информацию о кривизне дороги.

Метод ```slidewindow()``` принимает изображение в качестве входных данных и возвращает изображение со скользящими окнами, а также x-координату центра дороги, которая является важным индикатором кривизны полосы.

Этот файл Python также содержит класс WindowRange, который представляет собой определенную прямоугольную область изображения, называемую "окном".

## Участники проекта
<center>
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/jo49973477>">
        <img src="https://github.com/jo49973477.png" width="150px;" alt="Yeongyoo Jo"/>
        <br />
        <sub><b>Yeongyoo Jo</b></sub>
      </a>
      <br />
      <a href="https://github.com/jo49973477"><img src="https://img.shields.io/badge/GitHub-jo49973477-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/isragogreen">
        <img src="https://github.com/isragogreen.png" width="150px;" alt="Konstantin Tyhomyrov"/>
        <br />
        <sub><b>Konstantin Tyhomyrov</b></sub>
      </a>
      <br />
      <a href="https://github.com/isragogreen"><img src="https://img.shields.io/badge/GitHub-isragogreen-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
    <td align="center">
      <a href="https://github.com/indiks">
        <img src="https://github.com/indiks.png" width="150px;" alt="Sergey Indik"/>
        <br />
        <sub><b>Sergey Indik</b></sub>
      </a>
      <br />
      <a href="https://github.com/indiks"><img src="https://img.shields.io/badge/GitHub-indiks-blue?logo=github" alt="GitHub Badge" /></a>
      <br />
    </td>
  </tr>
</table>
</center>

