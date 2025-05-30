# 🧠 App de Análisis de Sentimiento (Frontend + Backend en EC2)

Esta aplicación móvil permite analizar el sentimiento (positivo, neutro o negativo) de un texto ingresado por el usuario. El análisis es realizado por un modelo con RNN desplegado en una API alojada en una instancia EC2 de AWS y una aplicación móvil.

---

## 🚀 Descripción General del Proyecto

- **Frontend móvil:** desarrollado en React Native, permite al usuario ingresar un texto y obtener un análisis emocional.

- **Backend (API):** construido con **FastAPI**, expone un endpoint `/predict/` que recibe un texto, lo traduce (si es necesario), lo procesa con un modelo RNN LSTM y devuelve la predicción.

- **Infraestructura:** desplegado en una instancia EC2 de Ubuntu con los servicios levantados usando `uvicorn`.

Requisitos del Proyecto

**Configuración del Backend en Ubuntu (AWS EC2)**
- Desarrollo del API con Flask o FastAPI
- Desarrollo del Frontend en VsC con React Naive
- Pruebas de Integración

**Backend (FastAPI)**
- AWS EC2 con Ubuntu
- FastAPI para el backend
- Modelo con RNN guardado con Joblib

**Frontend (React Native)**
- Windows 11 para el desarrollo
- React Native CLI
- Axios para hacer solicitudes HTTP

---

## Desarrollo del Back-end

1. En la consola de administración de AWS seleccione el servicio de EC2 (servidor virtual) o escriba en buscar.

![Vista de la App](images/Imagen1.png)

2. Ve a la opción para lanzar la instancia

![Vista de la App](images/Imagen2.png)

3. Lanza una instancia nueva

![Vista de la App](images/Imagen3.png)

4. Inicia una nueva instancia EC2 en AWS (elige Ubuntu como sistema operativo), puede dejar la imagen por defecto.

![Vista de la App](images/Imagen4.png)

5. Para este proyecto dado que el tamaño del modelo a descargar es grande necesitamos una maquina con más memoria y disco. con nuestra licencia tenemos permiso desde un micro lanzar hasta un T2.Large.

![Vista de la App](images/Imagen5.png)

6. Seleccione el par de claves ya creado, o cree uno nuevo (Uno de los dos, pero recuerde guardar esa llave que la puede necesitar, no la pierda)

![Vista de la App](images/Imagen6.png)

7. Habilite los puertos de shh, web y https, para este proyecto no lo vamos a usar no es necesario, pero si vas a publicar una web es requerido.

![Vista de la App](images/Imagen7.png)

8. Configure el almacenamiento. Este proyecto como se dijo requiere capacidad en disco. Aumente el disco a 16 GiB.

![Vista de la App](images/Imagen8.png)

9. Finalmente lance la instancia (no debe presentar error, si tiene error debe iniciar de nuevo). Si todo sale bien, por favor haga click en instancias en la parte superior.

![Vista de la App](images/Imagen9.png)

10. Vamos a seleccionar el servidor ec2 lanzado. Verificar la dirección IP pública y el DNS en el resumen de la instancia

![Vista de la App](images/Imagen10.png)

11. Debido a que vamos a lanzar un API rest debemos habilitar el puerto. Vamos al seguridad y luego vamos al grupo de seguridad

![Vista de la App](images/Imagen11.png)

12. Vamos a ir a Editar la regla de entrada

![Vista de la App](images/Imagen12.png)

13. Ahora vamos a agregar un regla de entrada para habilitar el puerto, recuerden poner IPV 4

![Vista de la App](images/Imagen13.png)

14. Abre un puerto en el grupo de seguridad (por ejemplo, puerto 8080) para permitir acceso a la API.

![Vista de la App](images/Imagen14.png)

15. Guardemos la regla de entrada.

![Vista de la App](images/Imagen15.png)

16. Ve nuevamente a instancias

![Vista de la App](images/Imagen16.png)

17. Vamos a conectar con la consola del servidor

![Vista de la App](images/Imagen17.png)


--

## Instalar dependencias en el Servidor EC2

Una vez dentro de tu instancia EC2, instalar las librerias y complementos como FastAPI y las dependencias necesarias para ello debes crear una carpeta en donde realizaras las instalaciones:

### Ver las carpetas
```bash
ls -la
```

### Ver la version de python
```bash
python3 -V
```

### Si se requiere, puede actualizar los paquetes
```bash
sudo apt update
```

### Instalar las siguientes instancias (images/imagen1)
![Vista de la App](images/Imagen18.png)

Si se requiere: Instalar pip y virtualenv
```bash
sudo apt update
```

Crear la carpeta del proyecto
```bash
sudo apt update
```

Accede a tu carpeta
```bash
sudo apt update
```

Crear y activar un entorno virtual
```bash
sudo apt update
```

Recuerda que en el prompt debe obersar que el env debe quedar activo
```bash
sudo apt update
```


## Crear la API FastAPI

Crea un archivo app.py en tu instancia EC2 para definir la API que servirá las predicciones.

```bash
nano app.py
```

![Vista de la App](images/Imagen24.png)


```bash
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator
import numpy as np
import pickle
import os
import random
import uvicorn

app = FastAPI()

# Paths locales en la instancia EC2
MODEL_PATH = "modelo_sentimiento.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Cargar modelo y tokenizador al iniciar
try:
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error cargando modelo o tokenizer: {str(e)}")
    raise e

# Respuestas personalizadas por clase
responses = {
    0: [
        "¡Parece que estás de buen ánimo! Sigue disfrutando tu día. 😊",
        "Tu mensaje refleja una actitud positiva. ¡Sigue así! 🌟",
        "Se nota optimismo en tus palabras. ¡Eso es genial! 💪"
    ],
    1: [
        "Tu mensaje parece ser neutral, sin una emoción fuerte asociada. 🤔",
        "No detecto un sentimiento marcado en tu mensaje. ¿Tienes algo en mente? 🧐",
        "Parece que es un comentario equilibrado, sin inclinación emocional. 🎭"
    ],
    2: [
        "Percibo que podrías estar sintiéndote mal. Si necesitas hablar, aquí estoy. 🖤",
        "Tu mensaje suena algo negativo. Espero que todo mejore pronto. 🌧️",
        "Parece que no estás en tu mejor día. Recuerda que todo pasa. 💙"
    ]
}

# Endpoint principal
@app.post("/predict/")
async def predict_sentiment(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "").strip()

        if not user_text:
            return JSONResponse(content={"error": "Texto vacío o no enviado."}, status_code=400)

        # Traducir de español a inglés
        translated_text = GoogleTranslator(source='es', target='en').translate(user_text)

        # Procesar entrada
        sequence = tokenizer.texts_to_sequences([translated_text])
        padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
        prediction = model.predict(padded)
        score = prediction[0][0]

        if score < 0.4:
            sentiment_class = 2  # Negativo
        elif score > 0.6:
            sentiment_class = 0  # Positivo
        else:
            sentiment_class = 1  # Neutro

        response_text = random.choice(responses.get(sentiment_class, ["Error: Clase fuera de rango."]))
        sentimiento_nombre = ["Positivo", "Neutro", "Negativo"][sentiment_class]

        return JSONResponse(content={
            "sentimiento": sentimiento_nombre,
            "respuesta": response_text,
            "score": float(score)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Para ejecutar directamente si se corre como script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Para salir del editor nano oprime CTRL-X y luego (Save modified buffer? ) escribe "Y" y (Save modified buffer? app.py) ENTER. puede verificar que archivo fue creado



## Ejecutar el Servidor FastAPI

Para ejecutar el servidor de FastAPI, usa Uvicorn:

```bash
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

![Vista de la App](images/Imagen25.png)

## Probar el servidor con un Curl desde el PowerShell del computador

Ejecutar el siguiente código para que envíe una predicción

```bash
Invoke-RestMethod -Uri http://3.82.114.41:8080/predict/ -Method POST -ContentType "application/json" -Body '{"text":"Estoy muy feliz con este proyecto"}'
```

---

## Desarrollo del Front-end



Node.js y npm:

## Paso 1: Verifica la instalación de Node.js y npm ejecutando en la terminal:
 ```bash

node -v
npm -v
 ```

Si no ves la versión de Node.js (al menos v18.x.x) o npm (v10.x.x), descarga e instala la versión LTS de Node.js desde aquí.

https://reactnative.dev/docs/set-up-your-environment

Va a requerir primero bajar 

https://chocolatey.org/install

Todos los pasos los puede verificar aqui

 https://youtu.be/nwXUXt_QqU8?si=dWjeavfLB06cz-bo


Verifica la instalación de Android Studio. Abre Android Studio y asegúrate de que el Android SDK y el Android Virtual Device (AVD) estén correctamente instalados, en el siguiente link puedes realizar la descarga.

https://developer.android.com/studio?hl=es-419&_gl=1*5t55h4*_up*MQ..&gclid=EAIaIQobChMIie2A3uCYiwMVJ7VaBR2njTbJEAAYASAAEgIdWvD_BwE&gclsrc=aw.ds

Dentro de Android Studio, ve a SDK Manager y asegúrate de que estén instaladas las siguientes herramientas:

Android SDK Platform 35.

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/SDK35.PNG?raw=true)

Intel x86 Atom System Image o Google APIs Intel x86 Atom System Image. (depende el procesador de tu maquina)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Intelx86.PNG?raw=true)

Android SDK Build Tools 35.0.0.

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/BuildTools.PNG?raw=true)

Si no tienes el AVD (Android Virtual Device), crea uno. Si tienes un dispositivo físico Android, puedes usarlo directamente conectándolo al PC a través de USB y habilitando la depuración USB en tu dispositivo.

Si no tienes el command-line tools, entra a la pagina de Android Studio 
https://developer.android.com/studio?gad_source=1&gclid=EAIaIQobChMIie2A3uCYiwMVJ7VaBR2njTbJEAAYASAAEgIdWvD_BwE&gclsrc=aw.ds&hl=es-419

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/comandTools.PNG?raw=true)

Una vez tienes el command-line tools debes extraerlo en el Android/SDK C:\Users\Smartcenter\AppData\Local\Android\Sdk

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/cmdline-tools.PNG?raw=true)

## Verficiación del NDK
* Abre Android Studio

Ve a: More Actions > SDK Manager

Haz clic en la pestaña SDK Tools

Marca la opción NDK (Side by side)

Si ya está marcada:

Desmárcala

Aplica cambios (esto desinstala)

* Luego vuelve a marcarla y aplica de nuevo (esto reinstala correctamente)

Asegúrate también de tener marcado:

CMake

Android SDK Command-line Tools (latest)

Reinicia Android Studio y tu terminal (cmd o Node.js Prompt)


## Variables de Entorno de Usuario:
Verifica que las variables de entorno estén correctamente configuradas, para ello accede a las variables de entorno desde el buscador de windows:

![image](https://github.com/user-attachments/assets/de660b10-e806-4229-af0f-a3a068cb5868)

Una vez estes ahi, dale click en variables de entorno

![image](https://github.com/user-attachments/assets/70aea713-c754-43e6-918a-938b5d81c4c5)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/android_Home.PNG?raw=true)


ANDROID_HOME debe apuntar a la carpeta de instalación del SDK de Android, el path de su cuenta o del sistema configure: Por ejemplo:
 ```plaintext

%LOCALAPPDATA%\Android\Sdk
%ANDROID_HOME%\tools\bin
%ANDROID_HOME%\emulator
%ANDROID_HOME%\platform-tools
%ANDROID_HOME%\tools
 ```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/android_Home.PNG?raw=true)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/sdk_entorno.PNG?raw=true)


##Asegúrate de que el emulador esté iniciado ANTES de correr run-android

Revisa que el dispositivo tenga una imagen compatible (por ejemplo, API 30 o superior)

Usa el emulador Pixel API 33 x86_64 (recomendado)


## Paso 2: Limpiar posibles residuos de instalaciones previas
Si has tenido problemas con instalaciones previas, es recomendable limpiar completamente las dependencias globales de npm y React Native.

Eliminar React Native CLI globalmente: Si tienes instalado react-native-cli globalmente, elimínalo:

 ```bash

npm uninstall -g react-native-cli
 ```
Eliminar la caché de npm: Borra la caché de npm para evitar problemas con dependencias:

 ```bash

npm cache clean --force
 ```

## Paso 3: Crear el Proyecto de React Native
Una vez que todo esté instalado y configurado correctamente, crea un nuevo proyecto de React Native con el siguiente comando:

Ejecutar directamente en Node.js Command Prompt en Administrador,
si prefieres no modificar las políticas de PowerShell, puedes usar el terminal proporcionado por Node.js:

Abre Node.js Command Prompt (generalmente instalado junto con Node.js).
Ejecuta tu comando:
 ```bash
npx @react-native-community/cli init imagenes
 ```
 (imagenes es el nombre del proyecto)


![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/React.PNG?raw=true)


Conectar tu dispositivo físico:
En adroide puedes configurar un    dispositivo virtual

En fisico:

Habilita Depuración por USB en tu dispositivo:
Ve a Configuración > Acerca del teléfono.
Toca varias veces en "Número de compilación" para habilitar el modo desarrollador.
Ve a Opciones de desarrollador y activa Depuración USB.
Conecta tu dispositivo a tu computadora con un cable USB.

Esto debería listar tu dispositivo.

Si no te llega a funcionar de este metodo, busca en google el modelo de tu celular y como activar el modo desarrollador

Accede a la carpeta de tu proyecto:

```bash
cd imagenes
```

Luego realiza una limpieza del cache:
```bash
npm cache clean --force
```

Ahora ejecuta el siguiente comando y veras la plantilla base de React Native
```bash
npx react-native run-android
```

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/plantilla.PNG?raw=true)

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Screenshot_2025-01-28-15-47-27-28_be78f1e3c60d0ba7def362c0a150a54c.jpg?raw=true)


## Paso 4: Instalar dependencias necesarias: 
Después de agregar el archivo App.js, asegúrate de que las dependencias que usas, como axios para HTTP y expo-image-picker, estén instaladas.
Instalaciones Requeridas: Asegúrate de haber instalado las dependencias necesarias:

```bash
npm install axios
```
```bash
npm install --save expo-image-picker
```
```bash
npm install react-native-permissions
```
```bash
npm install react-native-image-picker
```
```bash
npm install react-native-tts
```

## Paso 5: Crea el archivo app.tsx 
Cambia el archivo app.tsx y ejecuta estos comandos en el Visual Studio Code dentro del directorio de tu proyecto:

```bash
import React, { useState, useEffect } from 'react';
import { View, Text, TextInput, Button, StyleSheet, Alert, ScrollView, Platform, Image } from 'react-native';
import axios from 'axios';
import { launchCamera } from 'react-native-image-picker';
import { PermissionsAndroid, Platform as RNPlatform } from 'react-native'; 
import Tts from 'react-native-tts'; // Importa el paquete de texto a voz

const App = () => {
  const [ip, setIp] = useState('');
  const [puerto, setPuerto] = useState('');
  const [imagen, setImagen] = useState(null);
  const [respuesta, setRespuesta] = useState(null);

  // Función para solicitar permisos en Android
  const requestPermissions = async () => {
    if (Platform.OS === 'android') {
      try {
        const cameraPermission = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
          {
            title: 'Permiso para usar la cámara',
            message: 'La aplicación necesita acceso a la cámara para tomar fotos.',
            buttonNeutral: 'Pregúntame después',
            buttonNegative: 'Cancelar',
            buttonPositive: 'Aceptar',
          },
        );

        const storagePermission = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE,
          {
            title: 'Permiso para usar el almacenamiento',
            message: 'La aplicación necesita acceso al almacenamiento para guardar las fotos.',
            buttonNeutral: 'Pregúntame después',
            buttonNegative: 'Cancelar',
            buttonPositive: 'Aceptar',
          },
        );

        if (cameraPermission === PermissionsAndroid.RESULTS.GRANTED && storagePermission === PermissionsAndroid.RESULTS.GRANTED) {
          console.log('Permisos concedidos');
        } else {
          console.log('Permisos denegados');
        }
      } catch (err) {
        console.warn(err);
      }
    }
  };

  // Función para pedir permisos en iOS (si es necesario)
  const checkPermissionsForiOS = async () => {
    if (Platform.OS === 'ios') {
      const result = await launchCamera({ mediaType: 'photo' });
      if (result.errorCode) {
        Alert.alert('Error', 'No se pudieron obtener permisos para la cámara.');
      }
    }
  };

  // Llamar las funciones de permisos al inicio
  useEffect(() => {
    requestPermissions();
    checkPermissionsForiOS();
    Tts.setDefaultLanguage('es-ES'); 
    Tts.setDefaultRate(0.5); 
  }, []);

  const tomarFoto = async () => {
    const options = {
      mediaType: 'photo',
      cameraType: 'back',
      quality: 0.5,
    };

    launchCamera(options, (response) => {
      if (response.didCancel) {
        console.log('Usuario canceló la cámara');
      } else if (response.errorCode) {
        console.log('Error en la cámara: ', response.errorMessage);
      } else {
        const { uri } = response.assets[0];  // Usamos `assets[0]` ya que launchCamera retorna una matriz
        setImagen(uri);
      }
    });
  };

  const enviarImagen = async () => {
    if (!ip || !puerto) {
      Alert.alert('Error', 'Por favor ingresa la IP y el puerto del servidor.');
      return;
    }

    if (!imagen) {
      Alert.alert('Error', 'Por favor toma una foto antes de enviarla.');
      return;
    }

    const formData = new FormData();
    formData.append('file', {
      uri: imagen,
      type: 'image/jpeg',
      name: 'foto.jpg',
    });

    try {
      const response = await axios.post(`http://${ip}:${puerto}/predict/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setRespuesta(response.data.predictions);
      // Convertir la respuesta a voz
      speakResponse(response.data.predictions);
    } catch (error) {
      Alert.alert('Error', 'No se pudo enviar la imagen: ' + error.message);
    }
  };

  // Función para convertir la respuesta a voz
  const speakResponse = (predictions) => {
    if (predictions && predictions.length > 0) {
      const speechText = predictions.map(item => {
        return `La imagen se clasifica como ${item.class_name} con una probabilidad de ${(item.probability * 100).toFixed(2)}%.`;
      }).join(' '); // Unir todas las predicciones en un solo texto
      Tts.speak(speechText); // Convertir el texto a voz
    }
  };

  return (
    

    <View style={styles.container}>

      
      <View style={styles.header}>
        <Image source={require('./assets/logo.png')} style={styles.logo} />
        <Text style={styles.title}>Reconocimiento de Imagenes</Text>
      </View>

      <TextInput
        style={styles.input}
        placeholder="IP del servidor"
        value={ip}
        onChangeText={setIp}
        placeholderTextColor="white"
      />
      <TextInput
        style={styles.input}
        placeholder="Puerto del servidor"
        value={puerto}
        onChangeText={setPuerto}
        placeholderTextColor="white"
      />
      <Button title="Tomar Foto" onPress={tomarFoto} />
      <Button title="Clasificar Imagen" onPress={enviarImagen} />

      {respuesta && (
        <ScrollView style={styles.responseContainer}>
          <Text style={styles.responseTitle}>La imagen se puede clasificar en:</Text>
          {respuesta.map((item, index) => (
            <View key={index} style={styles.tableRow}>
              <Text style={styles.tableCell}>Class Name: {item.class_name}</Text>
              <Text style={styles.tableCell}>Probability: {(item.probability * 100).toFixed(2)}%</Text>
            </View>
          ))}
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  logo: {
    width: 50,
    height: 50,
    marginRight: 1,
    marginBottom:20,
    
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    textAlign: 'center',
    marginBottom: 20,
    color: 'black',
    flexWrap: 'wrap',  
    lineHeight: 30,    
    width: '80%',      
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 15,
    paddingLeft: 10,
    backgroundColor: 'black',
    color: 'white',
  },
  responseContainer: {
    marginTop: 20,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  responseTitle: {
    fontSize: 18,
    marginBottom: 10,
    fontWeight: 'bold',
  },
  tableRow: {
    marginBottom: 10,
  },
  tableCell: {
    fontSize: 16,
    color: 'black',
  },
});

export default App;

```
## Paso 6: Permisos en AndroidManifest.xml
Asegúrate de que los permisos para la cámara estén configurados en tu archivo AndroidManifest.xml: en C:\Users\USUARIO\imagenes\android\app\src\main\AndroidManifest.xml

```xml

<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.ACCESS_MEDIA_LOCATION"/>


```

## Paso 7: Crea la carpeta assets
En la raíz del proyecto crea una carpeta llamada assets en donde pongas la imagen del logo utilizada en el proyecto que podrás encontrar en los recursos de este github

![alt text](https://github.com/adiacla/FullStack-RNN/blob/main/Imagenes/Assets.PNG?raw=true)

## Paso 8: Ejecutar la App en el Emulador o en un Dispositivo Físico

Conectar un dispositivo Android físico y habilitar la Depuración USB en las Opciones de Desarrollador.

Puede listar si el dispoitivo está conectado
adb devices

listar los emuladores
emulator -list-avds



cd android
gradlew clean
cd ..

Ejecutar el Proyecto en un Dispositivo Físico:

```bash
npx react-native run-android
```
Emulador de Android: Si prefieres usar un emulador, puedes instalar Genymotion como alternativa al emulador de Android Studio:

Descargar Genymotion.
Configura el emulador con una imagen de Android y asegúrate de que adb detecte el emulador:
```bash
adb devices
```

---

## Despliegue Final

**Revisar Configuración de Seguridad en AWS**

Asegúrate de que el grupo de seguridad en AWS permita el tráfico en el puerto 8080 y que tu servidor sea accesible desde fuera de la red privada.

**Configuración de la manera de hacer el predict**

Debe estar bien configurado el código con la parte del post en la aplicación movil ya que utilizar otras funciones diferentes a la del app.py del código de la instancia no valdrá la función y no dará la predicción.

---
