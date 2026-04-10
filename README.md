# ArticleTopicRecognition

## Демо
http://176.57.79.250:8080

## Описание проекта

Сервис для автоматической классификации темы научных статей (arXiv) по их названию и аннотации.

Обучение проходило на https://www.kaggle.com/datasets/Cornell-University/arxiv

## Технологический стек
| Компонент | Технология |
|-----------|------------|
| **Модель** | `bert-base-cased` (Fine-tuned on arXiv dataset) |
| **Фреймворк** | PyTorch, HuggingFace `transformers` |
| **Инференс** | **NVIDIA TensorRT** (FP16) |
| **Web UI** | Streamlit |
| **Экспорт** | `optimum` (PyTorch -> ONNX -> TensorRT) |

## Hardware
Проект развернут на домашнем ПК

- **CPU:** AMD Ryzen 7 7800X3D
- **GPU:** NVIDIA GeForce RTX 4070 Super (12GB VRAM) — *используется для TensorRT инференса*
- **Deployment:** Доступ из интернета через Port Forwarding (NAT) на роутере.


В задании требовалось реализовать классификатор и поднять веб-интерфейс. Я решил усложнить задачу в части **инфраструктуры и производительности инференса**, так как стандартный запуск модели в PyTorch (Eager Mode) слишком медленный для production-подобного сервиса.

### 1. Оптимизация инференса через TensorRT
Вместо стандартного запуска `model.forward()`, я реализовал полный пайплайн конвертации модели:
`PyTorch Checkpoint` -> `ONNX` -> `TensorRT Engine`.

### 3. Dynamic Shapes & Memory Management
При конвертации в TensorRT настроены **динамические размеры батча и длины последовательности** (`min/opt/max shapes`). Это позволяет эффективно использовать VRAM: модель не аллоцирует память под максимальную длину текста (512 токенов), если пришел короткий запрос, что снижает латентность аллокации.

Код:
- app.py                            # Streamlit интерфейс (UI + логика запросов)
- notebooks/01_train_and_eval.ipynb # Обучение, валидация, экспорт в ONNX
- src/inference.py                  # Обертка над TensorRT C-API для Python