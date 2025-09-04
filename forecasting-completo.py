# 🚀 SCRIPT PARA EJECUTAR TU FORECAST DE MATERIALES
# Copia y pega este código, luego ejecuta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# [Aquí iría toda la clase ForecastClienteMaterial que ya tienes]

def ejecutar_mi_forecast():
    """Función simple para ejecutar tu forecast"""
    
    print("🎯 SISTEMA DE PREDICCIÓN DE MATERIALES")
    print("="*50)
    
    # PASO 1: Configurar archivo
    print("\n📁 PASO 1: Configurar archivo de datos")
    archivo_csv = input("Ingresa el nombre de tu archivo CSV (ej: ventas_2025.csv): ").strip()
    
    # Si no ingresa nada, usar archivo por defecto
    if not archivo_csv:
        archivo_csv = "datos_ventas.csv"
        print(f"   Usando archivo por defecto: {archivo_csv}")
    
    # PASO 2: Configurar parámetros
    print("\n⚙️ PASO 2: Configurar parámetros")
    
    # Mínimo de meses con datos
    try:
        min_meses = int(input("Mínimo de meses con datos para predecir (recomendado: 3): ") or "3")
    except:
        min_meses = 3
    
    # Mínimo volumen total
    try:
        min_volumen = float(input("Mínimo volumen total en M2 (recomendado: 100): ") or "100")
    except:
        min_volumen = 100.0
    
    # Cuántas combinaciones procesar
    try:
        top_combinaciones = int(input("¿Cuántas combinaciones procesar? (recomendado: 50-200): ") or "100")
    except:
        top_combinaciones = 100
    
    print(f"\n✅ Configuración:")
    print(f"   • Archivo: {archivo_csv}")
    print(f"   • Mínimo meses: {min_meses}")
    print(f"   • Mínimo volumen: {min_volumen:,.0f} M2")
    print(f"   • Top combinaciones: {top_combinaciones}")
    
    # PASO 3: Ejecutar análisis
    print(f"\n🔄 PASO 3: Iniciando análisis...")
    
    try:
        # Crear instancia del sistema
        sistema = ForecastClienteMaterial(archivo_csv)
        
        # Pipeline completo
        print("   1/6 Cargando datos...")
        sistema.cargar_y_preparar_datos()
        
        print("   2/6 Identificando combinaciones predictibles...")
        sistema.identificar_combinaciones_predictibles(
            min_meses=min_meses, 
            min_cantidad_total=min_volumen
        )
        
        print("   3/6 Generando predicciones...")
        sistema.ejecutar_forecast_completo(
            top_combinaciones=top_combinaciones, 
            meses_forecast=4
        )
        
        print("   4/6 Creando resumen ejecutivo...")
        sistema.mostrar_resumen_ejecutivo()
        
        print("   5/6 Generando reportes...")
        archivo_html = sistema.generar_reporte_detallado()
        archivo_excel = sistema.exportar_excel()
        
        print("   6/6 Creando gráficos...")
        archivo_grafico = sistema.generar_grafico_tendencias()
        
        # PASO 4: Mostrar resultados
        print(f"\n🎉 ¡PREDICCIONES COMPLETADAS!")
        print("="*50)
        print(f"📊 ARCHIVOS GENERADOS:")
        print(f"   • {archivo_html} (Reporte web completo)")
        print(f"   • {archivo_excel} (Datos para Excel)")
        print(f"   • {archivo_grafico} (Gráficos de tendencias)")
        
        print(f"\n📈 PRÓXIMOS PASOS:")
        print(f"   1. Abre {archivo_html} en tu navegador para ver el reporte completo")
        print(f"   2. Usa {archivo_excel} para análisis adicionales")
        print(f"   3. Revisa {archivo_grafico} para ver las tendencias")
        print(f"   4. Enfócate en predicciones con MAPE <50% (color verde y amarillo)")
        
        # Mostrar algunas predicciones clave
        if sistema.predicciones_detalladas:
            print(f"\n🔝 TOP 5 PREDICCIONES MÁS IMPORTANTES:")
            print("-" * 80)
            print(f"{'Cliente':<25} | {'Material':<20} | {'Sep':<8} | {'Oct':<8} | {'Nov':<8} | {'Dic':<8} | {'Total'}")
            print("-" * 80)
            
            for i, pred in enumerate(sistema.predicciones_detalladas[:5]):
                cliente_short = pred['cliente'][:23] + ".." if len(pred['cliente']) > 23 else pred['cliente']
                material_short = pred['material'][:18] + ".." if len(pred['material']) > 18 else pred['material']
                
                preds = pred['predicciones']
                sep_pred = f"{preds[0]['prediccion']:,.0f}" if len(preds) > 0 else "0"
                oct_pred = f"{preds[1]['prediccion']:,.0f}" if len(preds) > 1 else "0" 
                nov_pred = f"{preds[2]['prediccion']:,.0f}" if len(preds) > 2 else "0"
                dic_pred = f"{preds[3]['prediccion']:,.0f}" if len(preds) > 3 else "0"
                total_pred = f"{pred['total_predicho']:,.0f}"
                
                print(f"{cliente_short:<25} | {material_short:<20} | {sep_pred:<8} | {oct_pred:<8} | {nov_pred:<8} | {dic_pred:<8} | {total_pred}")
        
        return sistema
        
    except FileNotFoundError:
        print(f"❌ ERROR: No se encontró el archivo '{archivo_csv}'")
        print("   Verifica que el archivo esté en la misma carpeta que este script")
        return None
    except Exception as e:
        print(f"❌ ERROR durante la ejecución: {str(e)}")
        print("   Verifica el formato de tu archivo CSV")
        return None


def verificar_formato_csv(archivo_csv):
    """Función para verificar que tu CSV tenga el formato correcto"""
    
    print(f"\n🔍 VERIFICANDO FORMATO DE: {archivo_csv}")
    print("-" * 40)
    
    try:
        # Leer primeras 5 filas
        df_sample = pd.read_csv(archivo_csv, delimiter=';', encoding='utf-8', nrows=5)
        
        print(f"✅ Archivo encontrado!")
        print(f"📊 Columnas encontradas: {list(df_sample.columns)}")
        print(f"📝 Primeras 3 filas:")
        print(df_sample.head(3).to_string())
        
        # Verificar columnas requeridas
        columnas_requeridas = ['cliente', 'Material', 'cantidadComprada', 'sku', 'Año', 'Mes', 'Día']
        columnas_faltantes = []
        
        for col in columnas_requeridas:
            if col not in df_sample.columns:
                columnas_faltantes.append(col)
        
        if columnas_faltantes:
            print(f"\n⚠️ COLUMNAS FALTANTES: {columnas_faltantes}")
            print("   Tu archivo debe tener exactamente estas columnas:")
            for col in columnas_requeridas:
                print(f"     • {col}")
        else:
            print(f"\n✅ ¡Formato correcto! Puedes ejecutar el forecast.")
            
        return len(columnas_faltantes) == 0
        
    except FileNotFoundError:
        print(f"❌ Archivo no encontrado: {archivo_csv}")
        return False
    except Exception as e:
        print(f"❌ Error al leer archivo: {str(e)}")
        return False


def main_interactivo():
    """Función principal con menú interactivo"""
    
    print("🎯 SISTEMA DE PREDICCIÓN DE MATERIALES")
    print("=" * 50)
    print("Opciones:")
    print("1. Verificar formato de mi archivo CSV")
    print("2. Ejecutar predicciones completas")