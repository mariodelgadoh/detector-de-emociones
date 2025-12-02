import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from PIL import Image, ImageTk, ImageDraw
from mtcnn import MTCNN
import threading
from collections import deque
import mediapipe as mp

class SistemaDeteccionEmociones:
    def __init__(self, ruta_imagenes="imagenes"):
        # Directorio para almacenar datos
        self.data_dir = "datos_emociones"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Ruta para imágenes de personajes
        self.ruta_imagenes = ruta_imagenes
        os.makedirs(ruta_imagenes, exist_ok=True)
        
        # Inicializar detectores y modelos
        print("Cargando modelos de detección de emociones...")
        self.detector = MTCNN()
        
        # Inicializar MediaPipe para landmarks faciales
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Puntos clave específicos para emociones
        self.puntos_clave = {
            # Labios
            'labio_superior_centro': 13,
            'labio_inferior_centro': 14,
            'comisura_derecha': 61,
            'comisura_izquierda': 291,
            
            # Cejas
            'ceja_izq_interior': 105,
            'ceja_izq_exterior': 66,
            'ceja_der_interior': 334,
            'ceja_der_exterior': 296,
            
            # Ojos
            'ojo_izq_superior': 159,
            'ojo_izq_inferior': 145,
            'ojo_der_superior': 386,
            'ojo_der_inferior': 374,
            
            # Nariz
            'nariz_punta': 1,
            'nariz_base': 2,
            
            # Para lengua (nuevos puntos)
            'menton': 152,
            'labio_inferior_centro_abajo': 17
        }
        
        # Definir emociones principales estilo Intensamente
        self.emociones = {
            'ALEGRIA': {  # Felicidad - Alegría
                'color': (255, 223, 0),      # Amarillo brillante
                'tk_color': "#FFDF00",
                'descripcion': 'Sonrisa brillante! Emoción positiva y energética',
                'personaje': 'Alegría',
                'imagen': None
            },
            'TRISTEZA': {  # Tristeza
                'color': (66, 135, 245),     # Azul suave
                'tk_color': "#4287F5",
                'descripcion': 'Expresión melancólica, comisuras hacia abajo',
                'personaje': 'Tristeza',
                'imagen': None
            },
            'FURIA': {     # Enojo - Furia
                'color': (255, 50, 50),      # Rojo intenso
                'tk_color': "#FF3232",
                'descripcion': 'Cejas fruncidas, expresión intensa',
                'personaje': 'Furia',
                'imagen': None
            },
            'TEMOR': {     # Miedo - Temor
                'color': (180, 70, 230),     # Morado
                'tk_color': "#B446E6",
                'descripcion': 'Ojos abiertos, expresión de sorpresa/miedo',
                'personaje': 'Temor',
                'imagen': None
            },
            'DESAGRADO': { # Disgusto - Desagrado
                'color': (50, 200, 50),      # Verde
                'tk_color': "#32C832",
                'descripcion': 'Lengua fuera, ojos cerrados, expresión de rechazo',
                'personaje': 'Desagrado',
                'imagen': None
            }
        }
        
        # Cargar imágenes de los personajes
        self.cargar_imagenes_personajes()
        
        # Historial para suavizado
        self.historial_emociones = deque(maxlen=5)
        
        # Métricas
        self.metricas = {
            'total_detecciones': 0,
            'emociones_detectadas': {},
            'confianza_promedio': 0,
            'historial_detecciones': []
        }
        
        for emocion in self.emociones.keys():
            self.metricas['emociones_detectadas'][emocion] = 0
        
        # Valores de referencia para normalización
        self.referencias = {
            'boca_ancho_neutral': 70,
            'cejas_distancia_neutral': 85,
            'ojos_altura_neutral': 12,
        }
        
        self.primera_deteccion = True
    
    def cargar_imagenes_personajes(self):
        """Carga las imágenes de los personajes desde archivos PNG"""
        for emocion in self.emociones.keys():
            nombre_archivo = f"{emocion.lower()}.png"
            ruta_completa = os.path.join(self.ruta_imagenes, nombre_archivo)
            
            try:
                if os.path.exists(ruta_completa):
                    imagen = Image.open(ruta_completa)
                    # Redimensionar sin hacer circular (cuadradas)
                    imagen = imagen.resize((60, 60), Image.Resampling.LANCZOS)
                    self.emociones[emocion]['imagen'] = imagen
                    print(f"Imagen cargada: {nombre_archivo}")
                else:
                    # Crear imagen cuadrada de relleno con color
                    print(f"Creando imagen de relleno para: {emocion}")
                    imagen = Image.new('RGB', (60, 60), self.emociones[emocion]['tk_color'])
                    self.emociones[emocion]['imagen'] = imagen
            except Exception as e:
                print(f"Error cargando imagen para {emocion}: {e}")
                # Crear imagen cuadrada de relleno en caso de error
                imagen = Image.new('RGB', (60, 60), self.emociones[emocion]['tk_color'])
                self.emociones[emocion]['imagen'] = imagen
    
    def obtener_imagen_personaje(self, emocion, tamaño=(60, 60)):
        """Obtiene la imagen del personaje redimensionada"""
        if emocion in self.emociones and self.emociones[emocion]['imagen']:
            return self.emociones[emocion]['imagen'].copy()
        return None
    
    def obtener_color_tkinter(self, emocion):
        """Obtiene el color en formato Tkinter para una emoción"""
        return self.emociones.get(emocion, {}).get('tk_color', "#ffffff")
    
    def obtener_color_opencv(self, emocion):
        """Obtiene el color en formato OpenCV para una emoción"""
        return self.emociones.get(emocion, {}).get('color', (200, 200, 200))
    
    def obtener_personaje(self, emocion):
        """Obtiene el nombre del personaje para una emoción"""
        return self.emociones.get(emocion, {}).get('personaje', 'No Detectado')
    
    def extraer_rostro(self, frame):
        """Detecta y extrae el rostro de un frame"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detecciones = self.detector.detect_faces(frame_rgb)
            
            if len(detecciones) == 0:
                return None, None
            
            deteccion = max(detecciones, key=lambda x: x['confidence'])
            
            if deteccion['confidence'] < 0.7:
                return None, None
            
            x, y, ancho, alto = deteccion['box']
            x, y = max(0, x), max(0, y)
            
            if x + ancho > frame_rgb.shape[1]:
                ancho = frame_rgb.shape[1] - x
            if y + alto > frame_rgb.shape[0]:
                alto = frame_rgb.shape[0] - y
            
            rostro = frame_rgb[y:y+alto, x:x+ancho]
            
            if rostro.size == 0 or rostro.shape[0] < 20 or rostro.shape[1] < 20:
                return None, None
            
            return rostro, (x, y, ancho, alto)
        except Exception as e:
            print(f"Error en extraer_rostro: {e}")
            return None, None
    
    def obtener_landmarks(self, frame):
        """Obtiene landmarks faciales de MediaPipe"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            landmarks = {}
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks[idx] = (int(landmark.x * w), int(landmark.y * h))
            
            return landmarks
            
        except Exception as e:
            print(f"Error en obtener_landmarks: {e}")
            return None
    
    def analizar_expresion(self, landmarks):
        """Analiza la expresión facial basada en landmarks"""
        if not landmarks:
            return None
        
        resultados = {}
        
        try:
            # 1. ANÁLISIS DE BOCA (sonrisa/tristeza)
            if all(p in landmarks for p in [self.puntos_clave['comisura_derecha'], 
                                          self.puntos_clave['comisura_izquierda'],
                                          self.puntos_clave['nariz_punta']]):
                comisura_der = landmarks[self.puntos_clave['comisura_derecha']]
                comisura_izq = landmarks[self.puntos_clave['comisura_izquierda']]
                nariz = landmarks[self.puntos_clave['nariz_punta']]
                
                # Ancho de boca
                ancho_boca = np.linalg.norm(np.array(comisura_der) - np.array(comisura_izq))
                resultados['ancho_boca'] = ancho_boca
                
                # Posición de comisuras respecto a nariz
                altura_comisuras = (comisura_der[1] + comisura_izq[1]) / 2
                diferencia_nariz_comisuras = altura_comisuras - nariz[1]
                resultados['posicion_comisuras'] = diferencia_nariz_comisuras
                
                # Altura de boca (apertura) - importante para desagrado (lengua fuera)
                if all(p in landmarks for p in [self.puntos_clave['labio_superior_centro'],
                                              self.puntos_clave['labio_inferior_centro'],
                                              self.puntos_clave['menton']]):
                    labio_sup = landmarks[self.puntos_clave['labio_superior_centro']]
                    labio_inf = landmarks[self.puntos_clave['labio_inferior_centro']]
                    menton = landmarks[self.puntos_clave['menton']]
                    
                    altura_boca = abs(labio_sup[1] - labio_inf[1])
                    resultados['altura_boca'] = altura_boca
                    
                    # Distancia labio inferior a mentón (para detectar lengua fuera)
                    distancia_labio_menton = abs(labio_inf[1] - menton[1])
                    resultados['distancia_labio_menton'] = distancia_labio_menton
            
            # 2. ANÁLISIS DE CEJAS (enojo/sorpresa)
            if all(p in landmarks for p in [self.puntos_clave['ceja_izq_interior'],
                                          self.puntos_clave['ceja_izq_exterior'],
                                          self.puntos_clave['ceja_der_interior'],
                                          self.puntos_clave['ceja_der_exterior']]):
                ceja_izq_int = landmarks[self.puntos_clave['ceja_izq_interior']]
                ceja_izq_ext = landmarks[self.puntos_clave['ceja_izq_exterior']]
                ceja_der_int = landmarks[self.puntos_clave['ceja_der_interior']]
                ceja_der_ext = landmarks[self.puntos_clave['ceja_der_exterior']]
                
                # Distancia entre cejas (fruncimiento)
                distancia_cejas = np.linalg.norm(np.array(ceja_izq_int) - np.array(ceja_der_int))
                resultados['distancia_cejas'] = distancia_cejas
                
                # Inclinación de cejas
                inclinacion_izq = ceja_izq_int[1] - ceja_izq_ext[1]
                inclinacion_der = ceja_der_int[1] - ceja_der_ext[1]
                resultados['inclinacion_cejas'] = (inclinacion_izq + inclinacion_der) / 2
            
            # 3. ANÁLISIS DE OJOS (sorpresa/miedo - TEMOR / cerrados para DESAGRADO)
            if all(p in landmarks for p in [self.puntos_clave['ojo_izq_superior'],
                                          self.puntos_clave['ojo_izq_inferior'],
                                          self.puntos_clave['ojo_der_superior'],
                                          self.puntos_clave['ojo_der_inferior']]):
                ojo_izq_sup = landmarks[self.puntos_clave['ojo_izq_superior']]
                ojo_izq_inf = landmarks[self.puntos_clave['ojo_izq_inferior']]
                ojo_der_sup = landmarks[self.puntos_clave['ojo_der_superior']]
                ojo_der_inf = landmarks[self.puntos_clave['ojo_der_inferior']]
                
                altura_ojo_izq = abs(ojo_izq_sup[1] - ojo_izq_inf[1])
                altura_ojo_der = abs(ojo_der_sup[1] - ojo_der_inf[1])
                resultados['altura_ojos'] = (altura_ojo_izq + altura_ojo_der) / 2
                
                # Ancho de ojos (para detectar si están cerrados)
                ancho_ojo_izq = abs(landmarks[33][0] - landmarks[133][0]) if 33 in landmarks and 133 in landmarks else 0
                ancho_ojo_der = abs(landmarks[362][0] - landmarks[263][0]) if 362 in landmarks and 263 in landmarks else 0
                resultados['ancho_ojos'] = (ancho_ojo_izq + ancho_ojo_der) / 2
            
            # Actualizar referencias en la primera detección
            if self.primera_deteccion and 'ancho_boca' in resultados and 'distancia_cejas' in resultados:
                self.referencias['boca_ancho_neutral'] = resultados['ancho_boca']
                self.referencias['cejas_distancia_neutral'] = resultados['distancia_cejas']
                if 'altura_ojos' in resultados:
                    self.referencias['ojos_altura_neutral'] = resultados['altura_ojos']
                self.primera_deteccion = False
            
            return resultados
            
        except Exception as e:
            print(f"Error en analizar_expresion: {e}")
            return None
    
    def detectar_emocion(self, frame, bbox=None):
        """Detecta la emoción predominante en el rostro"""
        try:
            # Obtener landmarks
            landmarks = self.obtener_landmarks(frame)
            if not landmarks:
                return "NO_DETECTADO", 0.0, landmarks
            
            # Analizar expresión
            analisis = self.analizar_expresion(landmarks)
            if not analisis:
                return "NO_DETECTADO", 0.0, landmarks
            
            # Puntajes iniciales
            puntajes = {
                'ALEGRIA': 0,
                'TRISTEZA': 0,
                'FURIA': 0,
                'TEMOR': 0,
                'DESAGRADO': 0
            }
            
            # 1. DETECCIÓN DE ALEGRÍA (SONRISA)
            if 'ancho_boca' in analisis and 'posicion_comisuras' in analisis:
                ancho_boca = analisis['ancho_boca']
                posicion_comisuras = analisis['posicion_comisuras']
                
                referencia_boca = self.referencias.get('boca_ancho_neutral', 70)
                factor_ancho = min(max((ancho_boca - referencia_boca) / 30, 0), 2.0)
                
                if posicion_comisuras < -5:
                    factor_comisuras = abs(posicion_comisuras) * 0.5
                else:
                    factor_comisuras = 0
                
                puntajes['ALEGRIA'] = min(30 + (factor_ancho * 30) + factor_comisuras, 100)
            
            # 2. DETECCIÓN DE TRISTEZA
            if 'posicion_comisuras' in analisis and 'inclinacion_cejas' in analisis:
                posicion_comisuras = analisis['posicion_comisuras']
                inclinacion_cejas = analisis.get('inclinacion_cejas', 0)
                
                if posicion_comisuras > 10:
                    factor_comisuras = min(posicion_comisuras * 0.8, 40)
                else:
                    factor_comisuras = 0
                
                if inclinacion_cejas > 3:
                    factor_cejas = min(inclinacion_cejas * 5, 30)
                else:
                    factor_cejas = 0
                
                puntajes['TRISTEZA'] = min(20 + factor_comisuras + factor_cejas, 100)
            
            # 3. DETECCIÓN DE FURIA
            if 'distancia_cejas' in analisis:
                distancia_cejas = analisis['distancia_cejas']
                referencia_cejas = self.referencias.get('cejas_distancia_neutral', 85)
                diferencia = referencia_cejas - distancia_cejas
                
                if diferencia > 15:
                    puntajes['FURIA'] = min(40 + (diferencia * 2), 90)
                elif diferencia > 8:
                    puntajes['FURIA'] = min(30 + (diferencia * 1.5), 70)
                elif 'ancho_boca' in analisis:
                    ancho_boca = analisis['ancho_boca']
                    if ancho_boca < self.referencias.get('boca_ancho_neutral', 70) * 0.8:
                        puntajes['FURIA'] = max(puntajes['FURIA'], 40)
            
            # 4. DETECCIÓN DE TEMOR
            if 'altura_ojos' in analisis and 'altura_boca' in analisis:
                altura_ojos = analisis['altura_ojos']
                altura_boca = analisis['altura_boca']
                referencia_ojos = self.referencias.get('ojos_altura_neutral', 12)
                
                factor_ojos = min(max((altura_ojos - referencia_ojos) / 5, 0), 3.0)
                
                if altura_boca > 25:
                    factor_boca = min((altura_boca - 20) / 10, 2.0)
                else:
                    factor_boca = 0
                
                puntajes['TEMOR'] = min(30 + (factor_ojos * 20) + (factor_boca * 20), 100)
            
            # 5. DETECCIÓN DE DESAGRADO (CON LENGUA FUERA Y OJOS CERRADOS)
            desagrado_score = 0
            
            # Detectar lengua fuera (boca muy abierta y distancia labio-mentón grande)
            if 'altura_boca' in analisis and 'distancia_labio_menton' in analisis:
                altura_boca = analisis['altura_boca']
                distancia_labio_menton = analisis['distancia_labio_menton']
                
                # Si la boca está muy abierta (más de 40 píxeles)
                if altura_boca > 40:
                    desagrado_score += 30
                    
                    # Si además hay mucha distancia entre labio inferior y mentón
                    if distancia_labio_menton > 35:
                        desagrado_score += 25
            
            # Detectar ojos cerrados
            if 'altura_ojos' in analisis and 'ancho_ojos' in analisis:
                altura_ojos = analisis['altura_ojos']
                ancho_ojos = analisis['ancho_ojos']
                
                referencia_ojos = self.referencias.get('ojos_altura_neutral', 12)
                
                # Si los ojos están casi cerrados (altura muy pequeña)
                if altura_ojos < referencia_ojos * 0.4:
                    desagrado_score += 25
                elif altura_ojos < referencia_ojos * 0.6:
                    desagrado_score += 15
            
            # Nariz arrugada (cejas bajas)
            if 'distancia_cejas' in analisis and 'inclinacion_cejas' in analisis:
                distancia_cejas = analisis['distancia_cejas']
                inclinacion_cejas = analisis['inclinacion_cejas']
                
                referencia_cejas = self.referencias.get('cejas_distancia_neutral', 85)
                
                # Cejas bajas y fruncidas
                if distancia_cejas < referencia_cejas * 0.9 and inclinacion_cejas > 2:
                    desagrado_score += 20
            
            puntajes['DESAGRADO'] = min(desagrado_score, 95)
            
            # DETERMINAR EMOCIÓN GANADORA
            emocion_ganadora = max(puntajes.items(), key=lambda x: x[1])
            
            if emocion_ganadora[1] < 35:
                emocion_final = "NO_DETECTADO"
                confianza = emocion_ganadora[1]
            else:
                emocion_final = emocion_ganadora[0]
                confianza = min(emocion_ganadora[1], 99.9)
            
            # Suavizar con historial
            self.historial_emociones.append(emocion_final)
            if len(self.historial_emociones) >= 3:
                from collections import Counter
                conteo = Counter(self.historial_emociones)
                emocion_final = conteo.most_common(1)[0][0]
            
            # Actualizar métricas
            self.actualizar_metricas(emocion_final, confianza, 0.1)
            
            return emocion_final, confianza, landmarks
            
        except Exception as e:
            print(f"Error en detectar_emocion: {e}")
            return "ERROR", 0.0, None
    
    def dibujar_resultados(self, frame, landmarks, emocion, confianza, bbox=None):
        """Dibuja resultados en el frame"""
        frame_con_dibujo = frame.copy()
        
        try:
            # Dibujar puntos clave
            if landmarks:
                puntos_importantes = [
                    self.puntos_clave['comisura_derecha'],
                    self.puntos_clave['comisura_izquierda'],
                    self.puntos_clave['ceja_izq_interior'],
                    self.puntos_clave['ceja_der_interior'],
                    self.puntos_clave['nariz_punta']
                ]
                
                for idx in puntos_importantes:
                    if idx in landmarks:
                        cv2.circle(frame_con_dibujo, landmarks[idx], 4, (255, 255, 255), -1)
                        cv2.circle(frame_con_dibujo, landmarks[idx], 6, (30, 30, 30), 1)
            
            # Dibujar bounding box
            if bbox:
                x, y, w, h = bbox
                color = self.obtener_color_opencv(emocion)
                
                cv2.rectangle(frame_con_dibujo, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(frame_con_dibujo, (x+2, y+2), (x+w-2, y+h-2), (255, 255, 255), 1)
                
                personaje = self.obtener_personaje(emocion)
                etiqueta = f"{personaje}: {confianza:.1f}%"
                
                (text_width, text_height), _ = cv2.getTextSize(
                    etiqueta, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
                )
                
                cv2.rectangle(frame_con_dibujo, 
                            (x, y - text_height - 15),
                            (x + text_width + 20, y),
                            color, -1)
                
                cv2.rectangle(frame_con_dibujo, 
                            (x, y - text_height - 15),
                            (x + text_width + 20, y),
                            (255, 255, 255), 1)
                
                cv2.putText(frame_con_dibujo, etiqueta,
                          (x + 10, y - 5),
                          cv2.FONT_HERSHEY_DUPLEX, 0.7,
                          (255, 255, 255), 2)
            
            return frame_con_dibujo
            
        except Exception as e:
            print(f"Error dibujando resultados: {e}")
            return frame
    
    def actualizar_metricas(self, emocion, confianza, tiempo):
        """Actualiza las métricas del sistema"""
        self.metricas['total_detecciones'] += 1
        
        if emocion in self.metricas['emociones_detectadas']:
            self.metricas['emociones_detectadas'][emocion] += 1
        
        if self.metricas['total_detecciones'] == 1:
            self.metricas['confianza_promedio'] = confianza
        else:
            factor = 1.0 / min(self.metricas['total_detecciones'], 50)
            self.metricas['confianza_promedio'] = (
                (1 - factor) * self.metricas['confianza_promedio'] + 
                factor * confianza
            )
        
        registro = {
            'emocion': emocion,
            'confianza': confianza,
            'tiempo': tiempo,
            'fecha_hora': datetime.now()
        }
        self.metricas['historial_detecciones'].append(registro)
        
        if len(self.metricas['historial_detecciones']) > 100:
            self.metricas['historial_detecciones'] = self.metricas['historial_detecciones'][-100:]
    
    def obtener_metricas_tiempo_real(self):
        """Obtiene métricas actualizadas"""
        metricas = self.metricas.copy()
        
        total = metricas['total_detecciones']
        if total > 0:
            porcentajes = {}
            for emocion, count in metricas['emociones_detectadas'].items():
                porcentajes[emocion] = (count / total) * 100
        else:
            porcentajes = {emocion: 0 for emocion in self.emociones.keys()}
        
        return {
            'total_detecciones': total,
            'confianza_promedio': metricas['confianza_promedio'],
            'emociones_porcentajes': porcentajes,
            'emociones_detectadas': metricas['emociones_detectadas'],
            'ultimas_detecciones': metricas['historial_detecciones'][-5:] if metricas['historial_detecciones'] else []
        }
    
    def generar_reporte_emociones(self):
        """Genera un reporte de análisis de emociones"""
        reporte = "="*70 + "\n"
        reporte += "              INFORME DE ANÁLISIS EMOCIONAL - INSIDE-9\n"
        reporte += "="*70 + "\n\n"
        
        metricas = self.obtener_metricas_tiempo_real()
        
        reporte += "ESTADÍSTICAS GENERALES:\n"
        reporte += "-" * 40 + "\n"
        reporte += f"• Total de análisis: {metricas['total_detecciones']}\n"
        reporte += f"• Confianza promedio: {metricas['confianza_promedio']:.2f}%\n"
        reporte += f"• Hora del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        reporte += "DISTRIBUCIÓN DE EMOCIONES:\n"
        reporte += "-" * 40 + "\n"
        for emocion, porcentaje in metricas['emociones_porcentajes'].items():
            count = metricas['emociones_detectadas'].get(emocion, 0)
            personaje = self.obtener_personaje(emocion)
            barra = "#" * int(porcentaje / 5)
            reporte += f"{personaje}: {count} veces ({porcentaje:.2f}%) {barra}\n"
        
        reporte += "\nÚLTIMAS DETECCIONES:\n"
        reporte += "-" * 40 + "\n"
        for deteccion in metricas['ultimas_detecciones']:
            fecha = deteccion['fecha_hora'].strftime("%H:%M:%S")
            personaje = self.obtener_personaje(deteccion['emocion'])
            reporte += f"[{fecha}] - {personaje} ({deteccion['confianza']:.1f}%)\n"
        
        reporte += "\n" + "="*70 + "\n"
        reporte += "   ¡Las emociones dan forma a nuestra experiencia!\n"
        reporte += "="*70 + "\n"
        
        return reporte
    
    def iniciar_camara(self, indice_camara=0):
        """Inicia la captura de la cámara"""
        self.cap = cv2.VideoCapture(indice_camara)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                print("No se pudo abrir la cámara")
                return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        return True
    
    def detener_camara(self):
        """Detiene la captura de la cámara"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def obtener_frame(self):
        """Obtiene un frame de la cámara"""
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None


class InterfazDeteccionEmociones:
    def __init__(self, root):
        self.root = root
        self.root.title("Inside-9: Cuarto de Control Emocional")
        self.root.geometry("1200x700")
        
        # Establecer fondo negro como en la imagen
        self.root.configure(bg='black')
        
        self.sistema_emociones = SistemaDeteccionEmociones()
        self.modo = "ninguno"
        self.video_activo = False
        self.id_actualizacion = None
        
        # Almacenar referencias a imágenes Tkinter
        self.imagenes_tk = {}
        
        self.configurar_estilo()
        self.configurar_interfaz()
        self.centrar_ventana()
        
    def centrar_ventana(self):
        """Centra la ventana en la pantalla"""
        self.root.update_idletasks()
        ancho = self.root.winfo_width()
        alto = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.root.winfo_screenheight() // 2) - (alto // 2)
        self.root.geometry(f'{ancho}x{alto}+{x}+{y}')
    
    def configurar_estilo(self):
        """Configura el estilo visual de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores principales basados en la imagen (fondo negro)
        bg_oscuro = 'black'
        bg_medio = '#111111'
        bg_claro = '#222222'
        color_primario = '#00b4d8'
        color_secundario = '#ff6b6b'
        color_texto = '#ffffff'
        color_subtitulo = '#90e0ef'
        
        # Configurar estilos
        style.configure('Titulo.TLabel',
                       background=bg_oscuro,
                       foreground=color_primario,
                       font=('Arial', 24, 'bold'))
        
        style.configure('Subtitulo.TLabel',
                       background=bg_oscuro,
                       foreground=color_subtitulo,
                       font=('Arial', 12))
        
        style.configure('Estado.TLabel',
                       background=bg_medio,
                       foreground='#00ff88',
                       font=('Arial', 11, 'bold'),
                       padding=5)
        
        style.configure('BotonPrimario.TButton',
                       background=color_primario,
                       foreground=color_texto,
                       font=('Arial', 11, 'bold'),
                       padding=10,
                       borderwidth=0)
        
        style.map('BotonPrimario.TButton',
                 background=[('active', '#0096c7'), ('pressed', '#0077b6')])
        
        style.configure('BotonSecundario.TButton',
                       background=color_secundario,
                       foreground=color_texto,
                       font=('Arial', 11, 'bold'),
                       padding=10,
                       borderwidth=0)
        
        style.configure('Marco.TFrame',
                       background=bg_medio,
                       relief='solid',
                       borderwidth=2)
        
        style.configure('TituloPanel.TLabelframe',
                       background=bg_oscuro,
                       foreground=color_primario,
                       font=('Arial', 12, 'bold'))
        
        style.configure('TituloPanel.TLabelframe.Label',
                       background=bg_oscuro,
                       foreground=color_primario)
        
    def crear_imagenes_tkinter(self):
        """Crea las imágenes de Tkinter para los personajes"""
        for emocion in self.sistema_emociones.emociones.keys():
            imagen_pil = self.sistema_emociones.obtener_imagen_personaje(emocion)
            if imagen_pil:
                self.imagenes_tk[emocion] = ImageTk.PhotoImage(imagen_pil)
    
    def configurar_interfaz(self):
        """Configura los elementos de la interfaz"""
        # Crear imágenes Tkinter
        self.crear_imagenes_tkinter()
        
        # Frame principal
        frame_principal = ttk.Frame(self.root, style='Marco.TFrame')
        frame_principal.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Configurar grid
        frame_principal.columnconfigure(0, weight=3)
        frame_principal.columnconfigure(1, weight=2)
        frame_principal.rowconfigure(1, weight=1)
        
        # ===== HEADER =====
        frame_header = ttk.Frame(frame_principal)
        frame_header.grid(row=0, column=0, columnspan=2, pady=(0, 20), sticky='ew')
        
        # Título principal
        lbl_titulo = ttk.Label(frame_header,
                              text="Inside-9: Cuarto de Control Emocional",
                              style='Titulo.TLabel')
        lbl_titulo.pack()
        
        # Subtítulo
        lbl_subtitulo = ttk.Label(frame_header,
                                 text="FOR NEW EMOTIONS - DETECCIÓN EN TIEMPO REAL",
                                 style='Subtitulo.TLabel')
        lbl_subtitulo.pack(pady=(5, 10))
        
        # Indicador de estado
        self.lbl_estado = ttk.Label(frame_header,
                                   text="SISTEMA LISTO - ESPERANDO CONEXIÓN",
                                   style='Estado.TLabel',
                                   relief='solid',
                                   borderwidth=1)
        self.lbl_estado.pack()
        
        # ===== PANEL IZQUIERDO: VIDEO Y CONTROLES =====
        frame_izquierdo = ttk.Frame(frame_principal, style='Marco.TFrame')
        frame_izquierdo.grid(row=1, column=0, padx=(0, 10), pady=(0, 10), sticky='nsew')
        frame_izquierdo.columnconfigure(0, weight=1)
        frame_izquierdo.rowconfigure(0, weight=1)
        frame_izquierdo.rowconfigure(1, weight=0)
        
        # Panel de video
        frame_video = ttk.LabelFrame(frame_izquierdo,
                                    text="PANEL DE VISUALIZACIÓN EMOCIONAL",
                                    style='TituloPanel.TLabelframe')
        frame_video.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        frame_video.columnconfigure(0, weight=1)
        frame_video.rowconfigure(0, weight=1)
        
        # Canvas para el video
        self.canvas_video = tk.Canvas(frame_video,
                                     width=640,
                                     height=480,
                                     bg='black',
                                     highlightthickness=0)
        self.canvas_video.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        # Mensaje inicial
        self.texto_sin_video = self.canvas_video.create_text(
            320, 240,
            text="CÁMARA DESCONECTADA\n\n"
                 "Presiona 'INICIAR ANÁLISIS'\n"
                 "para activar el sistema\n\n"
                 "Descubre tus emociones",
            font=('Arial', 14, 'bold'),
            fill='#00b4d8',
            justify='center'
        )
        
        # Panel de controles
        frame_controles = ttk.Frame(frame_izquierdo, style='Marco.TFrame')
        frame_controles.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 10))
        
        # Botones de control
        self.btn_iniciar = ttk.Button(frame_controles,
                                     text="INICIAR ANÁLISIS EMOCIONAL",
                                     command=self.iniciar_deteccion,
                                     style='BotonPrimario.TButton')
        self.btn_iniciar.pack(side='left', padx=10, pady=10, fill='x', expand=True)
        
        self.btn_detener = ttk.Button(frame_controles,
                                     text="DETENER ANÁLISIS",
                                     command=self.detener_camara,
                                     style='BotonSecundario.TButton',
                                     state='disabled')
        self.btn_detener.pack(side='right', padx=10, pady=10, fill='x', expand=True)
        
        # ===== PANEL DERECHO: ESTADÍSTICAS Y EMOCIONES =====
        frame_derecho = ttk.Frame(frame_principal, style='Marco.TFrame')
        frame_derecho.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky='nsew')
        frame_derecho.rowconfigure(0, weight=1)
        frame_derecho.rowconfigure(1, weight=0)
        
        # Panel de estadísticas
        frame_stats = ttk.LabelFrame(frame_derecho,
                                    text="TABLERO EMOCIONAL",
                                    style='TituloPanel.TLabelframe')
        frame_stats.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Contenedor para las emociones
        frame_emociones = ttk.Frame(frame_stats, style='Marco.TFrame')
        frame_emociones.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar las emociones
        self.labels_stats = {}
        self.canvas_barras = {}
        self.label_imagenes = {}
        
        for i, (emocion, info) in enumerate(self.sistema_emociones.emociones.items()):
            frame_emocion = ttk.Frame(frame_emociones, style='Marco.TFrame')
            frame_emocion.pack(fill='x', padx=5, pady=8)
            
            # Imagen del personaje (cuadrada)
            if emocion in self.imagenes_tk:
                lbl_img = tk.Label(frame_emocion,
                                 image=self.imagenes_tk[emocion],
                                 bg='#111111')
                lbl_img.pack(side='left', padx=(10, 15), pady=10)
                self.label_imagenes[emocion] = lbl_img
            
            # Frame para texto y barra
            frame_texto_barra = ttk.Frame(frame_emocion)
            frame_texto_barra.pack(side='left', fill='both', expand=True, padx=(0, 10))
            
            # Nombre y estadísticas
            personaje = info['personaje']
            color_tk = info['tk_color']
            
            lbl_nombre = tk.Label(frame_texto_barra,
                                text=personaje,
                                font=('Arial', 11, 'bold'),
                                bg='#111111',
                                fg=color_tk,
                                anchor='w')
            lbl_nombre.pack(fill='x', pady=(5, 2))
            
            lbl_stats = tk.Label(frame_texto_barra,
                               text="0 detecciones (0.0%)",
                               font=('Arial', 9),
                               bg='#111111',
                               fg='#ffffff',
                               anchor='w')
            lbl_stats.pack(fill='x', pady=(0, 5))
            self.labels_stats[emocion] = lbl_stats
            
            # Barra de progreso
            canvas_barra = tk.Canvas(frame_texto_barra,
                                    height=20,
                                    bg='#111111',
                                    highlightthickness=0)
            canvas_barra.pack(fill='x', pady=(0, 5))
            self.canvas_barras[emocion] = canvas_barra
        
        # Botón de reporte
        frame_boton = ttk.Frame(frame_derecho)
        frame_boton.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 10))
        
        self.btn_reporte = ttk.Button(frame_boton,
                                     text="GENERAR INFORME EMOCIONAL",
                                     command=self.generar_reporte,
                                     style='BotonPrimario.TButton')
        self.btn_reporte.pack(fill='x', pady=10)
        
        # Actualizar estadísticas iniciales
        self.actualizar_estadisticas()
    
    def actualizar_estadisticas(self):
        """Actualiza las estadísticas mostradas"""
        metricas = self.sistema_emociones.obtener_metricas_tiempo_real()
        
        for emocion, label in self.labels_stats.items():
            count = metricas['emociones_detectadas'].get(emocion, 0)
            porcentaje = metricas['emociones_porcentajes'].get(emocion, 0.0)
            
            if count == 0:
                texto = f"0 detecciones (0.0%)"
            else:
                texto = f"{count} detección{'es' if count != 1 else ''} ({porcentaje:.1f}%)"
            
            label.config(text=texto)
        
        # Actualizar barras de progreso
        self.actualizar_barras_progreso()
    
    def actualizar_barras_progreso(self):
        """Actualiza las barras de progreso de emociones"""
        metricas = self.sistema_emociones.obtener_metricas_tiempo_real()
        
        for emocion, canvas in self.canvas_barras.items():
            canvas.delete("all")
            
            porcentaje = metricas['emociones_porcentajes'].get(emocion, 0)
            color_tk = self.sistema_emociones.obtener_color_tkinter(emocion)
            
            # Tamaño de la barra
            ancho_barra = 200
            ancho_progreso = int(ancho_barra * (porcentaje / 100))
            
            # Dibujar fondo
            canvas.create_rectangle(0, 0, ancho_barra, 20,
                                  fill='#222222', outline='', width=0)
            
            # Dibujar progreso
            canvas.create_rectangle(0, 0, ancho_progreso, 20,
                                  fill=color_tk, outline='', width=0)
            
            # Borde
            canvas.create_rectangle(0, 0, ancho_barra, 20,
                                  outline='#00b4d8', width=1)
            
            # Texto del porcentaje
            if porcentaje > 0:
                texto_x = min(ancho_progreso - 25, ancho_barra - 25)
                if texto_x < 25:
                    texto_x = ancho_progreso + 25
                
                canvas.create_text(texto_x, 10,
                                 text=f"{porcentaje:.1f}%",
                                 fill='white' if porcentaje > 50 else 'black',
                                 font=('Arial', 8, 'bold'))
    
    def iniciar_deteccion(self):
        """Inicia la detección de emociones"""
        if self.sistema_emociones.iniciar_camara():
            self.modo = "deteccion"
            self.video_activo = True
            self.btn_iniciar.config(state='disabled')
            self.btn_detener.config(state='normal')
            
            self.lbl_estado.config(text="ANALIZANDO EMOCIONES - SISTEMA ACTIVO")
            
            # Actualizar canvas
            self.canvas_video.delete(self.texto_sin_video)
            self.actualizar_video()
        else:
            self.lbl_estado.config(text="ERROR DE CÁMARA - SISTEMA INACTIVO")
            messagebox.showerror("Error de Cámara",
                                "No se pudo acceder a la cámara web.\n\n"
                                "Por favor, verifica que:\n"
                                "• La cámara esté conectada\n"
                                "• No esté siendo usada por otra aplicación\n"
                                "• Los permisos de cámara estén habilitados")
    
    def detener_camara(self):
        """Detiene la cámara"""
        self.video_activo = False
        if self.id_actualizacion:
            self.root.after_cancel(self.id_actualizacion)
            self.id_actualizacion = None
        
        self.sistema_emociones.detener_camara()
        self.modo = "ninguno"
        self.btn_iniciar.config(state='normal')
        self.btn_detener.config(state='disabled')
        
        # Restaurar mensaje
        self.canvas_video.delete("all")
        self.texto_sin_video = self.canvas_video.create_text(
            320, 240,
            text="ANÁLISIS DETENIDO\n\n"
                 "Sistema en modo de espera\n\n"
                 "Presiona 'INICIAR ANÁLISIS'\n"
                 "para reactivar la detección\n\n"
                 "Tus emociones te esperan",
            font=('Arial', 14, 'bold'),
            fill='#00b4d8',
            justify='center'
        )
        
        self.lbl_estado.config(text="SISTEMA EN PAUSA - LISTO PARA REANUDAR")
    
    def actualizar_video(self):
        """Actualiza el feed de video con detección de emociones"""
        if not self.video_activo:
            return
        
        frame = self.sistema_emociones.obtener_frame()
        
        if frame is not None:
            # Detectar rostro
            rostro, bbox = self.sistema_emociones.extraer_rostro(frame)
            
            if rostro is not None:
                # Detectar emoción
                emocion, confianza, landmarks = self.sistema_emociones.detectar_emocion(frame)
                
                # Dibujar resultados
                frame_con_dibujo = self.sistema_emociones.dibujar_resultados(
                    frame, landmarks, emocion, confianza, bbox
                )
            else:
                frame_con_dibujo = frame
                cv2.putText(frame_con_dibujo, "BUSCANDO EMOCIONES...",
                          (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 105, 180), 2)
                cv2.putText(frame_con_dibujo, "Muestra tu rostro a la cámara",
                          (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (138, 43, 226), 1)
            
            # Marca de tiempo
            timestamp = datetime.now().strftime("Inside-9 • %H:%M:%S")
            cv2.putText(frame_con_dibujo, timestamp,
                       (10, frame_con_dibujo.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 216), 1)
            
            # Mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame_con_dibujo, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas_video.delete("all")
            self.canvas_video.create_image(320, 240, image=imgtk, anchor='center')
            self.canvas_video.imgtk = imgtk
            
            # Actualizar estadísticas periódicamente
            if self.sistema_emociones.metricas['total_detecciones'] % 5 == 0:
                self.actualizar_estadisticas()
        
        self.id_actualizacion = self.root.after(30, self.actualizar_video)
    
    def generar_reporte(self):
        """Genera y muestra un reporte de análisis"""
        reporte = self.sistema_emociones.generar_reporte_emociones()
        
        ventana_reporte = tk.Toplevel(self.root)
        ventana_reporte.title("Inside-9 - Informe de Análisis Emocional")
        ventana_reporte.geometry("600x700")
        ventana_reporte.configure(bg='black')
        ventana_reporte.transient(self.root)
        ventana_reporte.grab_set()
        
        # Centrar ventana de reporte
        ventana_reporte.update_idletasks()
        ancho = ventana_reporte.winfo_width()
        alto = ventana_reporte.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (ancho // 2)
        y = (self.root.winfo_screenheight() // 2) - (alto // 2)
        ventana_reporte.geometry(f'{ancho}x{alto}+{x}+{y}')
        
        # Frame principal
        frame_reporte = ttk.Frame(ventana_reporte)
        frame_reporte.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Título
        lbl_titulo_reporte = tk.Label(frame_reporte,
                                     text="INFORME DE ANÁLISIS EMOCIONAL",
                                     font=('Arial', 16, 'bold'),
                                     fg='#00b4d8',
                                     bg='black')
        lbl_titulo_reporte.pack(pady=(0, 10))
        
        # Área de texto
        texto_reporte = scrolledtext.ScrolledText(frame_reporte,
                                                 font=('Consolas', 10),
                                                 bg='#111111',
                                                 fg='#ffffff',
                                                 wrap='word',
                                                 padx=15,
                                                 pady=15)
        texto_reporte.pack(fill='both', expand=True, pady=(0, 20))
        texto_reporte.insert('1.0', reporte)
        texto_reporte.config(state='disabled')
        
        # Frame para botones
        frame_botones = ttk.Frame(frame_reporte)
        frame_botones.pack(fill='x')
        
        def guardar_reporte():
            ruta_archivo = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"),
                          ("Todos los archivos", "*.*")],
                initialfile=f"Inside9_Reporte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            if ruta_archivo:
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write(reporte)
                messagebox.showinfo("Reporte Guardado",
                                  f"Informe guardado exitosamente:\n{ruta_archivo}")
        
        btn_guardar = ttk.Button(frame_botones,
                                text="GUARDAR REPORTE",
                                command=guardar_reporte,
                                style='BotonPrimario.TButton')
        btn_guardar.pack(side='left', padx=5, fill='x', expand=True)
        
        btn_cerrar = ttk.Button(frame_botones,
                               text="CERRAR",
                               command=ventana_reporte.destroy,
                               style='BotonSecundario.TButton')
        btn_cerrar.pack(side='right', padx=5, fill='x', expand=True)
    
    def al_cerrar(self):
        """Maneja el cierre de la aplicación"""
        self.video_activo = False
        if self.id_actualizacion:
            self.root.after_cancel(self.id_actualizacion)
        self.sistema_emociones.detener_camara()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = InterfazDeteccionEmociones(root)
    root.protocol("WM_DELETE_WINDOW", app.al_cerrar)
    
    # Configurar icono si existe
    try:
        root.iconbitmap('intensamente.ico')
    except:
        pass
    
    root.mainloop()


if __name__ == "__main__":
    main()