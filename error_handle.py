from CTkMessagebox import CTkMessagebox

def error_handle(code, log=None, parent=None):
    match code:
        case 0:
            CTkMessagebox(title="Успешно", message="Успешно")
        case 1:
            CTkMessagebox(title="Ошибка", message="Не выбрано ни одного изображения")
        case 2:
            CTkMessagebox(title="Ошибка", message="Изображений недостаточно для реконструкции")
        case 3:
            CTkMessagebox(title="Ошибка", message=log)
        case 4:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле загрузки")
        case 5:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле проверки изображений")
        case 6:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле обработки изображений")
        case 7:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле преобразования изображений")
        case 8:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле выделения точек")
        case 9:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле сопоставления точек")
        case 10:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле триангуляции точек")
        case 11:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле уплотнения точек")
        case 12:
            CTkMessagebox(title="Ошибка", message="Ошибка в модуле построения поверхности")
        case 13:
            CTkMessagebox(title="Ошибка", message="Ошибка в функции преобразования меша в облако")
        case 14:
            CTkMessagebox(title="Ошибка", message="Изображения не предобработаны. Будут использованы изначальные изображения")
        case 15:
            CTkMessagebox(title="Ошибка", message="Изображения не обработаны. Вернитесь к предыдущему шагу")
        case 16:
            CTkMessagebox(title="Ошибка", message="Точки не выделены. Вернитесь на предыдущий шаг")
        case 17:
            CTkMessagebox(title="Ошибка", message="Точки не сопаставлены. Вернитесь на предыдущий шаг")
        case 18:
            CTkMessagebox(title="Ошибка", message="Модель не существует")
        case 19:
            CTkMessagebox(title="Ошибка", message=log)

        case 99:
            CTkMessagebox(title="Ошибка", message="Операция отменена пользователем")

        case 100: 
            CTkMessagebox(title="Ошибка", message="неизвестная ошибка")
