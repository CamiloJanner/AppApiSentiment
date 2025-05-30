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

![Vista de la App](images/imagen1.png)

2. Ve a la opción para lanzar la instancia

![Vista de la App](images/imagen1.png)

3. Lanza una istancia nueva

![Vista de la App](images/imagen1.png)

4. Inicia una nueva instancia EC2 en AWS (elige Ubuntu como sistema operativo), puede dejar la imagen por defecto.

![Vista de la App](images/imagen1.png)

5. Para este proyecto dado que el tamaño del modelo a descargar es grande necesitamos una maquina con más memoria y disco. con nuesra licencia tenemos permiso desde un micro lanzar hasta un T2.Large.

![Vista de la App](images/imagen1.png)

6. Seleccione el par de claves ya creado, o cree uno nuevo (Uno de los dos, pero recuerde guardar esa llave que la puede necesitar, no la pierda)

![Vista de la App](images/imagen1.png)

7. Habilite los puertos de shh, web y https, para este proyecto no lo vamos a usar no es necesario, pero si vas a publicar una web es requerido.

![Vista de la App](images/imagen1.png)

8. Configure el almacenamiento. Este proyecto como se dijo requere capacidad en disco. Aumente el disco a 16 GiB.

![Vista de la App](images/imagen1.png)

9. Finalmente lance la instancia (no debe presentar error, si tiene error debe iniciar de nuevo). Si todo sale bien, por favor haga click en instancias en la parte superior.

![Vista de la App](images/imagen1.png)

10. Dado que normalmente en la lista de instancias NO VE la nueva instancia lanzada por favor actualice la pagina Web o en ir a instancias.

![Vista de la App](images/imagen1.png)

11. Vamos a seleccionar el servidor ec2 lanzado.

![Vista de la App](images/imagen1.png)

12. Verificar la dirección IP pública y el DNS en el resumen de la instancia

![Vista de la App](images/imagen1.png)

13. Debido a que vamos a lanzar un API rest debemos habilitar el puerto. Vamos al seguridad

![Vista de la App](images/imagen1.png)

14. Vamos al grupo de seguridad

![Vista de la App](images/imagen1.png)

15. Vamos a ir a Editar la regla de entrada

![Vista de la App](images/imagen1.png)

16. Ahora vamos a agregar un regla de entrada para habilitar el puerto, recuerden poner IPV 4

![Vista de la App](images/imagen1.png)

17. Abre un puerto en el grupo de seguridad (por ejemplo, puerto 8080) para permitir acceso a la API.

![Vista de la App](images/imagen1.png)

18. Guardemos la regla de entrada.

![Vista de la App](images/imagen1.png)

19. Ve nuevamente a instancias

![Vista de la App](images/imagen1.png)

20. Vamos a conectar con la consola del servidor

![Vista de la App](images/imagen1.png)


---

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
![Vista de la App](images/imagen18.png)

---

## Desarrollo del Front-end


---

## 🔧 Paso a Paso del Desarrollo

### 1. Backend (API FastAPI en EC2)

#### 🛠️ Configuración inicial del entorno

```bash
# En tu instancia EC2 (Ubuntu)
sudo apt update && sudo apt install python3-pip
pip install fastapi uvicorn keras tensorflow numpy deep-translator


