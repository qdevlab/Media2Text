# Media2Text

Транскрибирует и диаризирует аудио/видео в текст через OpenAI, с таймкодами и нарезкой на сегменты.

Transcribes and diarizes audio/video into text via OpenAI, with timecodes and segment splitting.

***

### Описание

Media2Text - консольная утилита на Python, которая транскрибирует и диаризирует аудио/видео в текст. На вход подаётся каталог; утилита рекурсивно находит `.mp4`, нарезает звук на сегменты через ffmpeg, отправляет каждый сегмент в OpenAI (`gpt-4o-transcribe-diarize`) и собирает результат - чистый текст и текст с таймкодами - рядом с исходным файлом.

### Функционал

Каталог задаётся аргументом командной строки, язык - параметром `--language`. Обход рекурсивный; уже обработанные файлы (префикс `+`) и сегменты пропускаются.

На каждый файл создаётся отдельная временная папка, в неё ffmpeg режет звук на сегменты (по 300 секунд, с обнулением таймингов); после обработки папка удаляется, поэтому несколько файлов в одном каталоге не пересекаются. Каждый сегмент транскрибируется и диаризируется через OpenAI (`response_format=diarized_json`).

Рядом с исходником пишутся два файла: `<имя>__text.txt` (чистый текст) и `<имя>__text.timecodes.txt` (с таймкодами `[ЧЧ:ММ:СС]`). Таймкоды прореживаются: реплики одного спикера склеиваются в абзац, метка ставится не чаще заданного интервала, глобальное смещение учитывает границы сегментов. Отдельный детектор подсвечивает сегменты, где модель могла соскочить с целевого языка. Обработанный файл помечается префиксом `+`, ошибка на одном файле не останавливает остальные.

### Результаты

Реализован конвейер: каталог видео → нарезка → пакет вызовов OpenAI → транскрипт с таймкодами, без ручной обработки. Проверено на реальной записи встречи: нарезка на части, диаризация и вывод сверены с исходным файлом.

### Технологии

Python. OpenAI API (`gpt-4o-transcribe-diarize`, `diarized_json`). ffmpeg - нарезка звука. python-dotenv - конфигурация (ключ и пути в `.env`). Пути через `pathlib` - работает на Windows и Linux/macOS.

### Роль

Личная разработка: утилита целиком - обход каталога, нарезка, вызовы API, сборка текста и таймкодов, выгрузка.

### Ограничения

Вход - только `.mp4`. Нужны установленный ffmpeg и ключ OpenAI (транскрипция платная). Качество транскрипции и диаризации определяется моделью OpenAI; язык задаётся параметром (по умолчанию `en`), модель иногда уходит в другой язык - это подсвечивается, но не исправляется. Сегменты режутся по фиксированному времени.

---
### Overview

Media2Text is a Python console tool that transcribes and diarizes audio/video into text. It takes a directory, recursively finds `.mp4` files, splits the audio into segments with ffmpeg, sends each segment to OpenAI (`gpt-4o-transcribe-diarize`), and assembles the result - plain text and text with timecodes - next to the source file.

### Features

The directory is passed as a command-line argument, the language via `--language`. The scan is recursive; already-processed files (`+` prefix) and segments are skipped.

Each file gets its own temporary folder where ffmpeg splits the audio into segments (300 seconds each, with reset timestamps); the folder is deleted after processing, so several files in one directory do not collide. Each segment is transcribed and diarized through OpenAI (`response_format=diarized_json`).

Two files are written next to the source: `<name>__text.txt` (plain text) and `<name>__text.timecodes.txt` (with `[HH:MM:SS]` timecodes). Timecodes are thinned out: consecutive lines from one speaker are merged into a paragraph, a mark is printed no more than once per interval, and a global offset accounts for segment boundaries. A separate detector flags segments where the model may have drifted off the target language. The processed file is marked with a `+` prefix, and an error on one file does not stop the rest.

### Results

An end-to-end pipeline was implemented: a directory of videos → splitting → a batch of OpenAI calls → a transcript with timecodes, with no manual work. Verified on a real meeting recording: splitting, diarization and output were cross-checked against the source file.

### Technologies

Python. OpenAI API (`gpt-4o-transcribe-diarize`, `diarized_json`). ffmpeg for audio splitting. python-dotenv for configuration (key and paths in `.env`). Paths via `pathlib` - works on Windows and Linux/macOS.

### Role

Personal project: the whole tool - directory walking, splitting, API calls, assembling text and timecodes, export.

### Limitations

Input is `.mp4` only. Requires an installed ffmpeg and an OpenAI API key (transcription is paid). Transcription and diarization quality depend on the OpenAI model; the language is set by a parameter (default `en`), and the model occasionally drifts to another language - this is flagged, not corrected. Segments are split by a fixed duration.

***
