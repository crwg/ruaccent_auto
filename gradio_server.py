import gradio as gr
import onnxruntime
onnxruntime.set_default_logger_severity(3)
from ruaccent import RUAccent
import os
import json

# Глобальные переменные
accentizer = None
last_load_settings = None          # хранит кортеж параметров, с которыми загружена модель
loading_model = False              # флаг, чтобы не запускать повторную загрузку

# ----------------------------------------------------------------------
def select_folder_dialog(current_folder):
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return current_folder
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Выберите папку для моделей ruaccent")
        root.destroy()
    except Exception:
        folder = ""
    return folder if folder else current_folder

# ----------------------------------------------------------------------
def load_accentizer(workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode):
    global accentizer, last_load_settings, loading_model
    if loading_model:
        return "⏳ Модель уже загружается, подождите..."
    loading_model = True
    try:
        if not workdir or not workdir.strip():
            workdir = "./ruaccent_model"
        os.makedirs(workdir, exist_ok=True)

        if not isinstance(device, str):
            device = "CPU"
        device = device.upper()
        if device not in ("CPU", "CUDA"):
            device = "CPU"

        if device == "CUDA":
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" not in available_providers:
                loading_model = False
                return (
                    "❌ CUDA недоступна для onnxruntime.\n\n"
                    "У вас установлен `onnxruntime` (CPU). Для использования GPU выполните:\n"
                    "pip uninstall onnxruntime\n"
                    "pip install onnxruntime-gpu\n\n"
                    "После этого перезапустите приложение и выберите устройство CUDA."
                )

        if isinstance(use_dictionary, str):
            use_dictionary = use_dictionary.lower() in ('true', '1', 'yes', 'on')
        else:
            use_dictionary = bool(use_dictionary)

        if isinstance(tiny_mode, str):
            tiny_mode = tiny_mode.lower() in ('true', '1', 'yes', 'on')
        else:
            tiny_mode = bool(tiny_mode)

        custom = {}
        if custom_dict and custom_dict.strip():
            for pair in custom_dict.split(','):
                if ':' in pair:
                    word, stress = pair.strip().split(':', 1)
                    custom[word.strip()] = stress.strip()

        accentizer = RUAccent()
        accentizer.load(
            omograph_model_size=omograph_model_size,
            use_dictionary=use_dictionary,
            custom_dict=custom,
            device=device,
            workdir=workdir,
            tiny_mode=tiny_mode
        )
        # Запоминаем, с какими параметрами загружена модель
        last_load_settings = (workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode)

        return (f"✅ Модель '{omograph_model_size}' загружена. "
                f"Словарь: {'вкл' if use_dictionary else 'выкл'}, "
                f"tiny_mode: {'вкл' if tiny_mode else 'выкл'}. "
                f"Устройство: {device}. Папка: {workdir}")
    except Exception as e:
        return f"❌ Ошибка загрузки модели: {str(e)}"
    finally:
        loading_model = False

def ensure_model_loaded(workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode):
    """Проверяет, загружена ли модель с нужными параметрами, и загружает при необходимости."""
    global accentizer, last_load_settings, loading_model

    # Приводим параметры к каноническому виду для сравнения
    def normalize_bool(x):
        if isinstance(x, str):
            return x.lower() in ('true', '1', 'yes', 'on')
        return bool(x)

    use_dict_norm = normalize_bool(use_dictionary)
    tiny_norm = normalize_bool(tiny_mode)
    device_norm = device.upper() if isinstance(device, str) else "CPU"
    if device_norm not in ("CPU", "CUDA"):
        device_norm = "CPU"

    current_settings = (workdir, device_norm, omograph_model_size, use_dict_norm, custom_dict, tiny_norm)

    if accentizer is not None and last_load_settings == current_settings:
        # Модель уже загружена с теми же параметрами
        return True

    # Нужна загрузка
    load_accentizer(workdir, device_norm, omograph_model_size, use_dict_norm, custom_dict, tiny_norm)
    return accentizer is not None

# ----------------------------------------------------------------------
def process_text_safe_word_by_word(text, accentizer):
    """Обрабатывает каждое слово отдельно, проблемные слова пропускает без ударений."""
    import re
    # Разбиваем на слова и остальное (пробелы, знаки препинания)
    tokens = re.findall(r'(\w+|[^\w\s]+|\s+)', text)
    result = []
    for token in tokens:
        if re.match(r'^\w+$', token):  # это слово (буквы/цифры/подчёркивание)
            try:
                stressed = accentizer.process_all(token)
                result.append(stressed)
            except Exception:
                result.append(token)   # слово с ошибкой – без ударений
        else:
            result.append(token)       # пробелы, знаки препинания – без изменений
    return ''.join(result)

def process_text(input_text, workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode):
    if not input_text or not input_text.strip():
        return "⚠️ Введите текст."
    
    ensure_model_loaded(workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode)
    if accentizer is None:
        return "⚠️ Не удалось загрузить модель. Проверьте настройки."
    
    # Безопасная обработка: слова с ошибками останутся без ударений
    return process_text_safe_word_by_word(input_text, accentizer)
    
def process_file(file_obj, workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode):
    if file_obj is None:
        return "", "⚠️ Файл не выбран."
    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, f"✅ Файл '{os.path.basename(path)}' загружен."
    except Exception as e:
        return "", f"❌ Ошибка чтения: {str(e)}"

# ----------------------------------------------------------------------
def save_all_settings(workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode):
    use_dict_bool = bool(use_dictionary) if not isinstance(use_dictionary, str) else use_dictionary.lower() in ('true','1','yes','on')
    tiny_mode_bool = bool(tiny_mode) if not isinstance(tiny_mode, str) else tiny_mode.lower() in ('true','1','yes','on')
    if isinstance(device, bool):
        device = "CUDA" if device else "CPU"
    elif not isinstance(device, str):
        device = str(device)
    device = device.upper()
    if device not in ("CPU", "CUDA"):
        device = "CPU"

    settings = {
        "workdir": workdir,
        "device": device,
        "omograph_model_size": omograph_model_size,
        "use_dictionary": use_dict_bool,
        "custom_dict": custom_dict,
        "tiny_mode": tiny_mode_bool
    }
    return json.dumps(settings)

def load_all_settings(settings_json):
    if not settings_json or settings_json == "null":
        return ("./ruaccent_model", "CPU", "turbo3.1", True, "", False)
    try:
        s = json.loads(settings_json)
        device = s.get("device", "CPU").upper()
        if device not in ("CPU", "CUDA"):
            device = "CPU"

        use_dict = s.get("use_dictionary", True)
        if isinstance(use_dict, str):
            use_dict = use_dict.lower() in ('true','1','yes','on')
        else:
            use_dict = bool(use_dict)

        tiny_mode = s.get("tiny_mode", False)
        if isinstance(tiny_mode, str):
            tiny_mode = tiny_mode.lower() in ('true','1','yes','on')
        else:
            tiny_mode = bool(tiny_mode)

        omograph_model_size = s.get("omograph_model_size", "turbo3.1")

        return (
            s.get("workdir", "./ruaccent_model"),
            device,
            omograph_model_size,
            use_dict,
            s.get("custom_dict", ""),
            tiny_mode
        )
    except:
        return ("./ruaccent_model", "CPU", "turbo3.1", True, "", False)

# JavaScript для localStorage
save_js = """
function(s) { localStorage.setItem('ruaccent_settings', s); return s; }
"""
load_js = """
function() { return localStorage.getItem('ruaccent_settings'); }
"""

# ----------------------------------------------------------------------
with gr.Blocks(title="Расстановка ударений (ruaccent)") as demo:
    gr.Markdown("# 📖 Расстановка ударений в русском тексте")

    settings_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Параметры модели")
            with gr.Row():
                workdir = gr.Textbox(label="Папка для моделей", value="./ruaccent_model", scale=3)
                folder_btn = gr.Button("📂 Выбрать папку", scale=1, variant="secondary")
            device = gr.Radio(label="Устройство", choices=["CPU", "CUDA"], value="CPU")
            omograph_model_size = gr.Dropdown(
                label="Модель",
                choices=["turbo3.1", "big_poetry"],
                value="turbo3.1"
            )
            use_dictionary = gr.Radio(
                label="Словарь",
                choices=[("✅ Включён", True), ("❌ Выключен", False)],
                value=True
            )
            tiny_mode = gr.Radio(
                label="Tiny mode",
                choices=[("Включён", True), ("Выключен", False)],
                value=False
            )
            custom_dict = gr.Textbox(label="Пользовательский словарь", lines=2,
                                     placeholder="слово1:удар+ение1, слово2:удар+ение2")
            load_btn = gr.Button("🚀 Загрузить модель вручную", variant="primary")
            load_output = gr.Textbox(label="Статус загрузки", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Ввод текста")
            with gr.Tabs():
                with gr.TabItem("Ввести текст"):
                    input_text = gr.Textbox(label="Исходный текст", lines=10)
                with gr.TabItem("Загрузить файл (.txt, .md)"):
                    file_input = gr.File(label="Выберите файл", file_types=[".txt", ".md"])
                    file_status = gr.Textbox(label="Статус файла", interactive=False)
            submit_btn = gr.Button("🎯 Расставить ударения", variant="primary")
            output_text = gr.Textbox(label="Результат", lines=15, interactive=False)

    # ---- Сохранение настроек при любом изменении ----
    # Функция, которая сохраняет настройки и возвращает их для State и JS
    def save_and_return(*args):
        s = save_all_settings(*args)
        return s

    # Компоненты, изменение которых сохраняем
    settings_components = [workdir, device, omograph_model_size, use_dictionary, custom_dict, tiny_mode]

    # При изменении любого из них – сохраняем
    for comp in settings_components:
        comp.change(
            fn=save_and_return,
            inputs=settings_components,
            outputs=settings_state
        ).then(
            fn=None, inputs=settings_state, outputs=None, js=save_js
        )

    # Кнопка выбора папки: сначала выбрать, потом сохранить
    folder_btn.click(
        fn=select_folder_dialog,
        inputs=workdir,
        outputs=workdir
    ).then(
        fn=save_and_return,
        inputs=settings_components,
        outputs=settings_state
    ).then(
        fn=None, inputs=settings_state, outputs=None, js=save_js
    ).then(
        fn=load_accentizer,
        inputs=settings_components,
        outputs=load_output
    )

    # Загрузка сохранённых настроек при старте
    demo.load(
        fn=None, inputs=None, outputs=settings_state, js=load_js
    ).then(
        fn=load_all_settings,
        inputs=settings_state,
        outputs=settings_components
    ).then(
        fn=load_accentizer,
        inputs=settings_components,
        outputs=load_output
    )

    # Ручная загрузка модели
    load_btn.click(
        fn=load_accentizer,
        inputs=settings_components,
        outputs=load_output
    )

    # Обработка текста – теперь с ленивой загрузкой модели
    submit_btn.click(
        fn=process_text,
        inputs=[input_text] + settings_components,
        outputs=output_text
    )
    file_input.change(
        fn=process_file,
        inputs=[file_input] + settings_components,
        outputs=[input_text, file_status]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())