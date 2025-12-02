import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
from PIL import Image, ImageTk
from mtcnn import MTCNN
import threading
from collections import deque
import mediapipe as mp

class SistemaDeteccionEmociones:
    def __init__(self):
        # Directorio para almacenar datos
        self.data_dir = "datos_emociones"
        os.makedirs(self.data_dir, exist_ok=True)
        
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
            'nariz_base': 2
        }
        
        # Definir emociones principales
        self.emociones = {
            'FELIZ': {
                'color': (0, 255, 0),      # Verde
                'tk_color': "#00ff00",
                'descripcion': 'Sonrisa visible, comisuras hacia arriba'
            },
            'TRISTE': {
                'color': (255, 0, 0),      # Azul
                'tk_color': "#0000ff",
                'descripcion': 'Comisuras hacia abajo, cejas inclinadas'
            },
            'ENOJADO': {
                'color': (0, 0, 255),      # Rojo
                'tk_color': "#ff0000",
                'descripcion': 'Cejas fruncidas, labios tensos'
            },
            'SORPRENDIDO': {
                'color': (0, 255, 255),    # Amarillo
                'tk_color': "#ffff00",
                'descripcion': 'Ojos abiertos, boca entreabierta'
            },
            'NEUTRAL': {
                'color': (200, 200, 200),  # Gris
                'tk_color': "#c8c8c8",
                'descripcion': 'Expresión relajada, normal'
            }
        }
        
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
        
        # Valores de referencia para normalización (se ajustarán automáticamente)
        self.referencias = {
            'boca_ancho_neutral': 70,      # Se ajustará
            'cejas_distancia_neutral': 85, # Se ajustará
            'ojos_altura_neutral': 12,     # Se ajustará
        }
        
        self.primera_deteccion = True
    
    def obtener_color_tkinter(self, emocion):
        """Obtiene el color en formato Tkinter para una emoción"""
        return self.emociones.get(emocion, {}).get('tk_color', "#ffffff")
    
    def obtener_color_opencv(self, emocion):
        """Obtiene el color en formato OpenCV para una emoción"""
        return self.emociones.get(emocion, {}).get('color', (200, 200, 200))
    
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
                
                # Altura de boca (apertura)
                if all(p in landmarks for p in [self.puntos_clave['labio_superior_centro'],
                                              self.puntos_clave['labio_inferior_centro']]):
                    labio_sup = landmarks[self.puntos_clave['labio_superior_centro']]
                    labio_inf = landmarks[self.puntos_clave['labio_inferior_centro']]
                    altura_boca = abs(labio_sup[1] - labio_inf[1])
                    resultados['altura_boca'] = altura_boca
            
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
                inclinacion_izq = ceja_izq_int[1] - ceja_izq_ext[1]  # Positivo = interior más bajo
                inclinacion_der = ceja_der_int[1] - ceja_der_ext[1]  # Positivo = interior más bajo
                resultados['inclinacion_cejas'] = (inclinacion_izq + inclinacion_der) / 2
            
            # 3. ANÁLISIS DE OJOS (sorpresa)
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
        """Detecta la emoción predominante en el rostro - VERSIÓN SIMPLIFICADA"""
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
                'FELIZ': 0,
                'TRISTE': 0,
                'ENOJADO': 0,
                'SORPRENDIDO': 0,
                'NEUTRAL': 50  # Puntaje base más alto
            }
            
            # 1. DETECCIÓN DE FELICIDAD (SONRISA)
            if 'ancho_boca' in analisis and 'posicion_comisuras' in analisis:
                ancho_boca = analisis['ancho_boca']
                posicion_comisuras = analisis['posicion_comisuras']
                
                # Sonrisa: boca ancha Y comisuras hacia arriba (respecto a nariz)
                referencia_boca = self.referencias.get('boca_ancho_neutral', 70)
                
                # Factor de ancho de boca (mientras más ancha, más feliz)
                factor_ancho = min(max((ancho_boca - referencia_boca) / 30, 0), 2.0)
                
                # Factor de posición de comisuras (mientras más negativa, más hacia arriba)
                if posicion_comisuras < -5:  # Comisuras claramente arriba de la nariz
                    factor_comisuras = abs(posicion_comisuras) * 0.5
                else:
                    factor_comisuras = 0
                
                puntajes['FELIZ'] = min(30 + (factor_ancho * 30) + factor_comisuras, 100)
            
            # 2. DETECCIÓN DE TRISTEZA
            if 'posicion_comisuras' in analisis and 'inclinacion_cejas' in analisis:
                posicion_comisuras = analisis['posicion_comisuras']
                inclinacion_cejas = analisis.get('inclinacion_cejas', 0)
                
                # Tristeza: comisuras hacia abajo Y cejas inclinadas
                if posicion_comisuras > 10:  # Comisuras claramente abajo de la nariz
                    factor_comisuras = min(posicion_comisuras * 0.8, 40)
                else:
                    factor_comisuras = 0
                
                # Cejas inclinadas (interior más bajo que exterior)
                if inclinacion_cejas > 3:  # Interior claramente más bajo
                    factor_cejas = min(inclinacion_cejas * 5, 30)
                else:
                    factor_cejas = 0
                
                puntajes['TRISTE'] = min(20 + factor_comisuras + factor_cejas, 100)
            
            # 3. DETECCIÓN DE ENOJO - ¡SIMPLIFICADO!
            if 'distancia_cejas' in analisis:
                distancia_cejas = analisis['distancia_cejas']
                referencia_cejas = self.referencias.get('cejas_distancia_neutral', 85)
                
                # Enojo: cejas muy juntas (fruncidas)
                diferencia = referencia_cejas - distancia_cejas
                
                if diferencia > 15:  # Cejas muy juntas
                    puntajes['ENOJADO'] = min(40 + (diferencia * 2), 90)
                elif diferencia > 8:  # Cejas algo juntas
                    puntajes['ENOJADO'] = min(30 + (diferencia * 1.5), 70)
                elif 'ancho_boca' in analisis:
                    # Boca tensa (poco ancha) también indica enojo
                    ancho_boca = analisis['ancho_boca']
                    if ancho_boca < self.referencias.get('boca_ancho_neutral', 70) * 0.8:
                        puntajes['ENOJADO'] = max(puntajes['ENOJADO'], 40)
            
            # 4. DETECCIÓN DE SORPRESA
            if 'altura_ojos' in analisis and 'altura_boca' in analisis:
                altura_ojos = analisis['altura_ojos']
                altura_boca = analisis['altura_boca']
                referencia_ojos = self.referencias.get('ojos_altura_neutral', 12)
                
                # Sorpresa: ojos muy abiertos Y boca abierta
                factor_ojos = min(max((altura_ojos - referencia_ojos) / 5, 0), 3.0)
                
                # Boca abierta (altura > 20 normalmente indica sorpresa)
                if altura_boca > 25:
                    factor_boca = min((altura_boca - 20) / 10, 2.0)
                else:
                    factor_boca = 0
                
                puntajes['SORPRENDIDO'] = min(30 + (factor_ojos * 20) + (factor_boca * 20), 100)
            
            # 5. AJUSTE DE NEUTRAL
            # NEUTRAL es alto por defecto, se reduce si hay emociones fuertes
            emocion_fuerte = max(puntajes.values())
            if emocion_fuerte > 60:
                # Si hay una emoción fuerte, reducir NEUTRAL
                puntajes['NEUTRAL'] = max(20, 100 - emocion_fuerte)
            
            # 6. DETERMINAR EMOCIÓN GANADORA
            emocion_ganadora = max(puntajes.items(), key=lambda x: x[1])
            
            # Umbral mínimo para considerar detección
            if emocion_ganadora[1] < 45 and emocion_ganadora[0] != 'NEUTRAL':
                # Si ninguna emoción es clara, forzar NEUTRAL
                emocion_final = "NEUTRAL"
                confianza = 60.0
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
            # Dibujar puntos clave si están disponibles
            if landmarks:
                # Puntos importantes
                puntos_importantes = [
                    self.puntos_clave['comisura_derecha'],
                    self.puntos_clave['comisura_izquierda'],
                    self.puntos_clave['ceja_izq_interior'],
                    self.puntos_clave['ceja_der_interior'],
                    self.puntos_clave['nariz_punta']
                ]
                
                for idx in puntos_importantes:
                    if idx in landmarks:
                        cv2.circle(frame_con_dibujo, landmarks[idx], 3, (0, 255, 255), -1)
            
            # Dibujar bounding box
            if bbox:
                x, y, w, h = bbox
                color = self.obtener_color_opencv(emocion)
                
                # Rectángulo
                cv2.rectangle(frame_con_dibujo, (x, y), (x+w, y+h), color, 2)
                
                # Etiqueta
                etiqueta = f"{emocion}: {confianza:.1f}%"
                (text_width, text_height), _ = cv2.getTextSize(
                    etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Fondo para texto
                cv2.rectangle(frame_con_dibujo, 
                            (x, y - text_height - 10),
                            (x + text_width + 10, y),
                            color, -1)
                
                # Texto
                cv2.putText(frame_con_dibujo, etiqueta,
                          (x + 5, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6,
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
        
        # Promedio móvil de confianza
        if self.metricas['total_detecciones'] == 1:
            self.metricas['confianza_promedio'] = confianza
        else:
            factor = 1.0 / min(self.metricas['total_detecciones'], 50)
            self.metricas['confianza_promedio'] = (
                (1 - factor) * self.metricas['confianza_promedio'] + 
                factor * confianza
            )
        
        # Historial
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
        reporte = "=== REPORTE DE ANÁLISIS DE EMOCIONES ===\n\n"
        
        metricas = self.obtener_metricas_tiempo_real()
        
        reporte += "ESTADÍSTICAS GENERALES:\n"
        reporte += f"Total de detecciones: {metricas['total_detecciones']}\n"
        reporte += f"Confianza promedio: {metricas['confianza_promedio']:.2f}%\n\n"
        
        reporte += "DISTRIBUCIÓN DE EMOCIONES:\n"
        for emocion, porcentaje in metricas['emociones_porcentajes'].items():
            count = metricas['emociones_detectadas'].get(emocion, 0)
            reporte += f"{emocion}: {count} veces ({porcentaje:.2f}%)\n"
        
        reporte += "\nÚLTIMAS DETECCIONES:\n"
        for deteccion in metricas['ultimas_detecciones']:
            fecha = deteccion['fecha_hora'].strftime("%H:%M:%S")
            reporte += f"{fecha} - {deteccion['emocion']} ({deteccion['confianza']:.1f}%)\n"
        
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
        self.root.title("Sistema de Detección de Emociones - EmotionAI")
        self.root.geometry("1280x920")
        self.root.configure(bg='#1a1a2e')
        
        self.sistema_emociones = SistemaDeteccionEmociones()
        self.modo = "ninguno"
        self.video_activo = False
        self.id_actualizacion = None
        
        self.configurar_estilo()
        self.configurar_interfaz()
        
    def configurar_estilo(self):
        """Configura el estilo visual de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        bg_dark = '#1a1a2e'
        bg_medium = '#16213e'
        accent_blue = '#00d4ff'
        
        style.configure('Modern.TFrame', background=bg_dark)
        style.configure('Card.TFrame', background=bg_medium)
        
        style.configure('Title.TLabel', 
                       background=bg_dark, 
                       foreground=accent_blue,
                       font=('Segoe UI', 20, 'bold'))
        
        style.configure('Primary.TButton',
                       background=accent_blue,
                       foreground='white',
                       font=('Segoe UI', 10, 'bold'),
                       padding=10)
        
    def configurar_interfaz(self):
        """Configura los elementos de la interfaz"""
        # Frame principal
        frame_principal = ttk.Frame(self.root, style='Modern.TFrame', padding="15")
        frame_principal.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame_principal.columnconfigure(0, weight=3)
        frame_principal.columnconfigure(1, weight=1)
        frame_principal.rowconfigure(1, weight=1)
        
        # Header
        frame_header = ttk.Frame(frame_principal, style='Modern.TFrame')
        frame_header.grid(row=0, column=0, columnspan=2, pady=(0, 15), sticky=(tk.W, tk.E))
        
        etiqueta_titulo = ttk.Label(frame_header, 
                                    text="EmotionAI - Detección de Emociones", 
                                    style='Title.TLabel')
        etiqueta_titulo.pack(side=tk.LEFT, padx=10)
        
        # Indicador de estado
        self.frame_estado = tk.Frame(frame_header, bg='#1a1a2e')
        self.frame_estado.pack(side=tk.RIGHT, padx=10)
        
        self.label_estado = tk.Label(self.frame_estado, 
                                     text="SISTEMA LISTO", 
                                     font=('Segoe UI', 10, 'bold'),
                                     fg='#00ff88',
                                     bg='#1a1a2e')
        self.label_estado.pack()
        
        # Frame de video
        self.frame_video = ttk.LabelFrame(frame_principal, 
                                         text="Análisis en Tiempo Real", 
                                         style='Modern.TLabelframe',
                                         padding="10")
        self.frame_video.grid(row=1, column=0, padx=(0, 10), pady=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        canvas_container = tk.Frame(self.frame_video, bg='#000000', relief='solid', borderwidth=2)
        canvas_container.pack(expand=True, fill='both')
        
        self.canvas_video = tk.Canvas(canvas_container, width=640, height=480, 
                                     bg='#0a0a0a', highlightthickness=0)
        self.canvas_video.pack(padx=2, pady=2)
        
        self.texto_sin_video = self.canvas_video.create_text(
            320, 240, 
            text="Cámara inactiva\nPresione 'Iniciar Detección' para comenzar",
            font=('Segoe UI', 14, 'bold'), 
            fill='#00d4ff',
            justify=tk.CENTER
        )
        
        # Panel de controles
        frame_controles = ttk.Frame(frame_principal, style='Modern.TFrame')
        frame_controles.grid(row=1, column=1, padx=(10, 0), pady=(0, 10), sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Sección de control
        frame_modo = ttk.LabelFrame(frame_controles, 
                                    text="Control del Sistema", 
                                    style='Modern.TLabelframe',
                                    padding="15")
        frame_modo.pack(fill='x', pady=(0, 10))
        
        self.boton_iniciar = ttk.Button(frame_modo, 
                                       text="Iniciar Detección", 
                                       command=self.iniciar_deteccion,
                                       style='Primary.TButton',
                                       width=25)
        self.boton_iniciar.pack(pady=5, fill='x')
        
        self.boton_detener = ttk.Button(frame_modo, 
                                       text="Detener Cámara", 
                                       command=self.detener_camara,
                                       style='Primary.TButton',
                                       state='disabled',
                                       width=25)
        self.boton_detener.pack(pady=5, fill='x')
        
        # Panel de estadísticas
        frame_stats = ttk.LabelFrame(frame_controles,
                                     text="Estadísticas de Emociones",
                                     style='Modern.TLabelframe',
                                     padding="15")
        frame_stats.pack(fill='both', expand=True, pady=(0, 10))
        
        self.labels_stats = {}
        for emocion in self.sistema_emociones.emociones.keys():
            frame_emocion = tk.Frame(frame_stats, bg='#16213e')
            frame_emocion.pack(fill='x', pady=3)
            
            color_tk = self.sistema_emociones.obtener_color_tkinter(emocion)
            canvas_color = tk.Canvas(frame_emocion, width=20, height=20, 
                                   bg=color_tk, highlightthickness=0)
            canvas_color.pack(side=tk.LEFT, padx=(0, 10))
            
            label = tk.Label(frame_emocion,
                           text=f"{emocion}: 0 (0.0%)",
                           font=('Segoe UI', 9),
                           bg='#16213e',
                           fg=color_tk,
                           anchor='w')
            label.pack(side=tk.LEFT, fill='x', expand=True)
            
            self.labels_stats[emocion] = label
        
        # Botón para reporte
        self.boton_reporte = ttk.Button(frame_controles,
                                       text="Generar Reporte",
                                       command=self.generar_reporte,
                                       style='Primary.TButton',
                                       width=25)
        self.boton_reporte.pack(fill='x', pady=10)
        
        # Panel de registro
        frame_registro = ttk.LabelFrame(frame_principal, 
                                       text="Registro de Detecciones", 
                                       style='Modern.TLabelframe',
                                       padding="10")
        frame_registro.grid(row=2, column=0, columnspan=2, padx=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        text_container = tk.Frame(frame_registro, bg='#0a0a0a', relief='solid', borderwidth=2)
        text_container.pack(fill='both', expand=True)
        
        self.texto_registro = scrolledtext.ScrolledText(text_container,
                                                        height=8,
                                                        font=('Consolas', 9),
                                                        bg='#0a0a0a',
                                                        fg='#00ff88',
                                                        insertbackground='#00ff88',
                                                        relief='flat')
        self.texto_registro.pack(padx=3, pady=3, fill='both', expand=True)
        
        # Mensaje inicial
        self.registrar("=" * 70, color='#00d4ff')
        self.registrar("SISTEMA DE DETECCIÓN DE EMOCIONES INICIADO", color='#00d4ff')
        self.registrar("=" * 70, color='#00d4ff')
        self.registrar("Emociones detectables:", color='#00ff88')
        for emocion, info in self.sistema_emociones.emociones.items():
            self.registrar(f"  • {emocion}: {info['descripcion']}", color='#00ff88')
        
        self.actualizar_estadisticas()
    
    def registrar(self, mensaje, color='#00ff88'):
        """Registra un mensaje en el área de texto"""
        marca_tiempo = datetime.now().strftime("%H:%M:%S")
        self.texto_registro.insert(tk.END, f"[{marca_tiempo}] {mensaje}\n")
        
        last_line_start = self.texto_registro.index("end-2c linestart")
        last_line_end = self.texto_registro.index("end-1c")
        tag_name = f"color_{color}"
        self.texto_registro.tag_config(tag_name, foreground=color)
        self.texto_registro.tag_add(tag_name, last_line_start, last_line_end)
        
        self.texto_registro.see(tk.END)
        self.root.update()
    
    def actualizar_estadisticas(self):
        """Actualiza las estadísticas mostradas"""
        metricas = self.sistema_emociones.obtener_metricas_tiempo_real()
        
        for emocion, label in self.labels_stats.items():
            count = metricas['emociones_detectadas'].get(emocion, 0)
            porcentaje = metricas['emociones_porcentajes'].get(emocion, 0.0)
            color_tk = self.sistema_emociones.obtener_color_tkinter(emocion)
            
            label.config(text=f"{emocion}: {count} ({porcentaje:.1f}%)", fg=color_tk)
    
    def iniciar_deteccion(self):
        """Inicia la detección de emociones"""
        self.registrar("Iniciando sistema de detección de emociones...", color='#00d4ff')
        
        if self.sistema_emociones.iniciar_camara():
            self.modo = "deteccion"
            self.video_activo = True
            self.boton_iniciar.config(state='disabled')
            self.boton_detener.config(state='normal')
            
            self.actualizar_estado("DETECTANDO EMOCIONES", '#00ff88')
            self.registrar("Sistema iniciado correctamente", color='#00ff88')
            self.registrar("Analizando expresiones faciales...", color='#00d4ff')
            self.actualizar_video()
        else:
            self.actualizar_estado("ERROR DE CÁMARA", '#ff0055')
            self.registrar("Error: No se pudo acceder a la cámara", color='#ff0055')
            messagebox.showerror("Error", "No se pudo acceder a la cámara web.")
    
    def detener_camara(self):
        """Detiene la cámara"""
        self.video_activo = False
        if self.id_actualizacion:
            self.root.after_cancel(self.id_actualizacion)
            self.id_actualizacion = None
        
        self.sistema_emociones.detener_camara()
        self.modo = "ninguno"
        self.boton_iniciar.config(state='normal')
        self.boton_detener.config(state='disabled')
        
        self.canvas_video.delete("all")
        self.texto_sin_video = self.canvas_video.create_text(
            320, 240, 
            text="Cámara detenida\nPresione 'Iniciar Detección' para reiniciar",
            font=('Segoe UI', 14, 'bold'), 
            fill='#00d4ff',
            justify=tk.CENTER
        )
        
        self.actualizar_estado("SISTEMA LISTO", '#00ff88')
        self.registrar("Detección detenida", color='#ffaa00')
    
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
                
                # Registrar emoción significativa
                if emocion != "NEUTRAL" and confianza > 60:
                    color_tk = self.sistema_emociones.obtener_color_tkinter(emocion)
                    self.registrar(f"Emoción detectada: {emocion} ({confianza:.1f}%)", 
                                 color=color_tk)
            else:
                frame_con_dibujo = frame
                cv2.putText(frame_con_dibujo, "Buscando rostro...", 
                          (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Marca de tiempo
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame_con_dibujo, timestamp, 
                       (10, frame_con_dibujo.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Mostrar en Tkinter
            frame_rgb = cv2.cvtColor(frame_con_dibujo, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas_video.delete("all")
            self.canvas_video.create_image(320, 240, image=imgtk, anchor=tk.CENTER)
            self.canvas_video.imgtk = imgtk
            
            # Actualizar estadísticas periódicamente
            if self.sistema_emociones.metricas['total_detecciones'] % 5 == 0:
                self.actualizar_estadisticas()
        
        self.id_actualizacion = self.root.after(50, self.actualizar_video)
    
    def generar_reporte(self):
        """Genera y muestra un reporte de análisis"""
        reporte = self.sistema_emociones.generar_reporte_emociones()
        
        ventana_reporte = tk.Toplevel(self.root)
        ventana_reporte.title("Reporte de Análisis de Emociones")
        ventana_reporte.geometry("600x700")
        ventana_reporte.configure(bg='#1a1a2e')
        
        texto_reporte = scrolledtext.ScrolledText(ventana_reporte,
                                                 font=('Consolas', 10),
                                                 bg='#0a0a0a',
                                                 fg='#00ff88',
                                                 wrap=tk.WORD)
        texto_reporte.pack(fill='both', expand=True, padx=10, pady=10)
        texto_reporte.insert('1.0', reporte)
        texto_reporte.config(state='disabled')
        
        frame_botones = tk.Frame(ventana_reporte, bg='#1a1a2e')
        frame_botones.pack(fill='x', padx=10, pady=10)
        
        def guardar_reporte():
            ruta_archivo = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
            )
            if ruta_archivo:
                with open(ruta_archivo, 'w', encoding='utf-8') as f:
                    f.write(reporte)
                messagebox.showinfo("Éxito", f"Reporte guardado en:\n{ruta_archivo}")
        
        ttk.Button(frame_botones, text="Guardar Reporte", 
                  command=guardar_reporte, style='Primary.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_botones, text="Cerrar", 
                  command=ventana_reporte.destroy).pack(side=tk.RIGHT, padx=5)
    
    def actualizar_estado(self, texto, color):
        """Actualiza el indicador de estado"""
        self.label_estado.config(text=texto, fg=color)
    
    def al_cerrar(self):
        """Maneja el cierre de la aplicación"""
        self.video_activo = False
        if self.id_actualizacion:
            self.root.after_cancel(self.id_actualizacion)
        self.sistema_emociones.detener_camara()
        
        self.registrar("=" * 70, color='#00d4ff')
        self.registrar("Sistema cerrado correctamente", color='#00d4ff')
        self.root.destroy()


def main():
    root = tk.Tk()
    app = InterfazDeteccionEmociones(root)
    root.protocol("WM_DELETE_WINDOW", app.al_cerrar)
    root.mainloop()


if __name__ == "__main__":
    main()