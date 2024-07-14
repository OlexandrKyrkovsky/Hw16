import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# Шляхи до моделей
model_path1 = '/Users/oleksandrkirkovskij/desktop/Hw16/model_cnn.h5'
model_path2 = '/Users/oleksandrkirkovskij/desktop/Hw16/model_vgg16.h5'

# Завантаження моделей
def load_model(model_type):
    custom_objects = {
        'TFOpLambda': tf.keras.layers.Lambda(lambda x: x)
    }
    if model_type == 'Convolutional Neural Network':
        model = tf.keras.models.load_model(model_path1, custom_objects=custom_objects)
    elif model_type == 'VGG16':
        model = tf.keras.models.load_model(model_path2, custom_objects=custom_objects)
    else:
        raise ValueError('Неправильний тип моделі')
    return model

# Варіанти моделей для вибору користувача
model_options = ['Convolutional Neural Network', 'VGG16']
selected_model = st.sidebar.selectbox('Оберіть модель', model_options)
model = load_model(selected_model)

# Назви класів (для моделі CNN)
class_names = ['Футболка/топ', 'Штани', 'Кофта', 'Сукня', 'Пальто', 'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевик']

# Функція для обробки зображення
def process_image(image, model_type):
    if model_type == 'Convolutional Neural Network':
        image = image.convert('L')  # Конвертування зображення у сіре
        image = image.resize((28, 28))
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)  # Додавання осі для каналу
        image = np.expand_dims(image, axis=0)
    elif model_type == 'VGG16':
        image = image.resize((224, 224))
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
    return image

# Основна частина додатку
def main():
    st.title('Класифікація зображень')
    
    uploaded_file = st.file_uploader("Оберіть зображення...", type=["jpg", "jpeg", "png"]) 
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Завантажене зображення.', use_column_width=True)
        
        processed_image = process_image(image, selected_model)
        prediction = model.predict(processed_image)
        
        if selected_model == 'Convolutional Neural Network':
            predicted_class = np.argmax(prediction)
            class_probabilities = prediction[0]
            
            st.write('Прогноз:')
            st.write(f'Прогнозований клас: {class_names[predicted_class]}')
            st.write('Ймовірності класів:')
            for i in range(len(class_names)):
                st.write(f'{class_names[i]}: {class_probabilities[i]}')
        elif selected_model == 'VGG16':
            decoded_predictions = decode_predictions(prediction, top=3)[0]
            st.write('Прогноз:')
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                st.write(f"{label}: {score:.2f}")
        
if __name__ == '__main__':
    main()
