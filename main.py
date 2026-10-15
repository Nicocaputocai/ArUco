import cv2
import numpy as np
import math
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constante física del ArUco (Medida real en mm verificada con calibre)
ARUCO_LADO_REAL_MM = 60.0 
# Factor para compensar la pérdida de perspectiva en cilindros
COEFICIENTE_PARALAJE = 1.33 

@app.post("/api/medir-diametro")
async def medir_diametro(
    foto: UploadFile = File(...),
    linea_x_izq: float = Form(...),
    linea_x_der: float = Form(...)
):
    try:
        contents = await foto.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Archivo de imagen inválido."})

        # Redimensionar la imagen a un máximo de 1000px
        alto_orig, ancho_orig = img.shape[:2]
        factor_escala = 1.0
        max_pixeles = 1000.0

        if max(alto_orig, ancho_orig) > max_pixeles:
            factor_escala = max_pixeles / max(alto_orig, ancho_orig)
            img_deteccion = cv2.resize(img, (int(ancho_orig * factor_escala), int(alto_orig * factor_escala)))
        else:
            img_deteccion = img.copy()

        # Parámetros tolerantes para el ArUco
        parametros = cv2.aruco.DetectorParameters()
        parametros.polygonalApproxAccuracyRate = 0.08
        parametros.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        parametros.adaptiveThreshWinSizeMin = 3
        parametros.adaptiveThreshWinSizeMax = 73
        parametros.adaptiveThreshWinSizeStep = 10

        diccionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(diccionario, parametros)
        esquinas, ids, rechazados = detector.detectMarkers(img_deteccion)

        mejor_area = 0
        esquinas_finales = None

        candidatos = []
        if esquinas is not None and len(esquinas) > 0:
            for eq in esquinas:
                candidatos.append(eq[0])
        
        if rechazados is not None and len(rechazados) > 0:
            for rech in rechazados:
                candidatos.append(rech[0])

        for pts in candidatos:
            ancho = (np.linalg.norm(pts[0] - pts[1]) + np.linalg.norm(pts[2] - pts[3])) / 2
            alto = (np.linalg.norm(pts[1] - pts[2]) + np.linalg.norm(pts[3] - pts[0])) / 2
            
            if ancho == 0 or alto == 0: 
                continue
                
            ratio = ancho / alto
            area = ancho * alto
            
            if 0.8 < ratio < 1.2 and area > 1500:
                if area > mejor_area:
                    mejor_area = area
                    esquinas_finales = pts

        if esquinas_finales is None:
            return JSONResponse(status_code=400, content={"error": "No se detectó el marco de la credencial en la foto."})

        # Volver a escala original
        esquinas_id1 = esquinas_finales / factor_escala 

        lados_pixeles = [
            np.linalg.norm(esquinas_id1[0] - esquinas_id1[1]), 
            np.linalg.norm(esquinas_id1[1] - esquinas_id1[2]), 
            np.linalg.norm(esquinas_id1[2] - esquinas_id1[3]), 
            np.linalg.norm(esquinas_id1[3] - esquinas_id1[0])  
        ]
        tamano_pixel_promedio = np.mean(lados_pixeles)

        escala_mm_por_pixel = float(ARUCO_LADO_REAL_MM / tamano_pixel_promedio)
        distancia_pixeles_tronco = float(abs(linea_x_der - linea_x_izq))
        
        diametro_optico_mm = distancia_pixeles_tronco * escala_mm_por_pixel
        diametro_optico_cm = float(diametro_optico_mm / 10.0)
        
        diametro_real_estimado = diametro_optico_cm * COEFICIENTE_PARALAJE

        return {
            "exito": True,
            "escala_mm_pixel": round(escala_mm_por_pixel, 4),
            "diametro_cm": round(diametro_real_estimado, 2)
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error interno: {str(e)}"})

@app.post("/api/medir-inclinacion")
async def medir_inclinacion(
    eje_x_base: float = Form(...),
    eje_y_base: float = Form(...),
    eje_x_alto: float = Form(...),
    eje_y_alto: float = Form(...)
):
    try:
        delta_x = eje_x_alto - eje_x_base
        delta_y = eje_y_base - eje_y_alto 
        
        if delta_y == 0:
            return JSONResponse(status_code=400, content={"error": "Línea horizontal."})

        angulo_radianes = math.atan2(delta_x, delta_y)
        inclinacion_grados = math.degrees(angulo_radianes)

        estado = "Recto"
        if abs(inclinacion_grados) > 5:
            direccion = "derecha" if inclinacion_grados > 0 else "izquierda"
            estado = f"Inclinado hacia la {direccion}"

        return {
            "exito": True,
            "inclinacion_grados": round(inclinacion_grados, 1),
            "estado": estado
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error interno: {str(e)}"})