# Обнаружения объектов YOLO и отслеживания объектов

#### Содержание
1. Вступление
2. Клонируйте репозиторий
3. Установка
    - [Linux](#3-Установка)
    - [Windows 10/11](#3-Установка) 
4. Примеры использования

## 1. Вступление

В данном репозитории реализованы несколько алгоритмов обнаружения и отслеживания в одном месте. Различные трэкеры, 
такие как "Byte Track", "Deep Sort" или "NorFair", могут быть интегрированы с различными версиями "YOLO" с минимальными строками кода.
Используется модели YOLO как в версиях `ONYX`, так и в версиях `Pitch`.

## 2. Клонируйте репозиторий

Перейдите в пустую папку по вашему выбору.

```git clone https://github.com/UmarNauruzov/people_tracker_CV```

## 3. Установка
<details open>
<summary>For Linux</summary>

```shell
python3 -m venv .env
source .env/bin/activate
#загрузка необходимых библиотек
pip install -r requirements.txt

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

```
</details>

<details>
<summary> For Windows 10/11</summary>

```shell
python -m venv .env
.env\Scripts\activate
#загрузка необходимых библиотек
pip install -r requirements.txt

#если cython_bbox из-за какой-то причины не установился используете следующую запись
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
</details>

## 4. Примеры использования
Для запуска программы необходимо создать виртуальную среду и загрузить туда все необходимые библиотеки указанные в `requirements.txt` и дополнительно библиотеки указанные выше.

При этом обращаю внимание, что может возникнуть проблема с загрузкой библиотеки lab, для решения проблемы, необходимо дополнительно установить пакет "Microsoft Visual C++ Build Tools", если он не
установлен на Вашем компьютере.


Запусти `main.py` чтобы протестировать трекер на `data/sample_videos/HallWayTracking/videos/001.avi` видео

```
python main.py data/sample_videos/HallWayTracking/videos/001.avi --draw_trails
```

Видео, которое показывает работы модели трэкера людей с отображением пройденного пути:

![hippo](https://github.com/UmarNauruzov/people_tracker_CV/blob/master/data/results/result_compressed1.gif)


### Run in `Google Colab`

 <a href="https://colab.research.google.com/drive/1tafZRbNl_BV65qWA-kbBdOt80J6GIe0V?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>