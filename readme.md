# Файл содержит описание к следующим скриптам в директории

## count_ced_for_points

Была добавлена нормализация по прямоугольнику лица, определяемому библиотекой dlib, изображения для лиц считываеются с папки gt_path

## face_point_detection_dlib

Скрипт предсказывает 68 ключевых точек для изображений лица, принимает следующие аргументы:

- __predictions_path__ путь до папки с картинками, лица которых надо предсказать
- __output_path__ путь до папки, где будут храниться .pts файлы с предсказанными точками, каждому изображению соответствует соотвутствующий .pts файл в данной директории
- __predictor_path__ путь до .dat файла с весами dlib модели

## face_alignment_utils

Ноутбук с предобработкой данных

## face_alignment_main

Рабочий ноутбук, в котором хранятся эксперименты для последующей реализации в скриптах
