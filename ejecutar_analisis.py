import pandas as pd
import numpy as np
from forecasting_system import comprehensive_forecast_enhanced

print("🔧 LIMPIEZA Y CORRECCIÓN DE DATOS")
print("="*50)

# Cargar datos
archivo = "zfacturacion bogota.csv"
df = pd.read_csv(archivo)

print(f"📊 Datos originales: {df.shape}")

# CORRECCIÓN 1: Limpiar cantidadComprada
print("\n🧹 Limpiando cantidadComprada...")
print(f"Tipo original: {df['cantidadComprada'].dtype}")
print("Valores únicos (muestra):", df['cantidadComprada'].unique()[:10])

# Convertir cantidadComprada a numérico
df['cantidadComprada'] = df['cantidadComprada'].astype(str).str.replace(',', '.')
df['cantidadComprada'] = pd.to_numeric(df['cantidadComprada'], errors='coerce')

# Eliminar registros con cantidades inválidas
df = df.dropna(subset=['cantidadComprada'])
df = df[df['cantidadComprada'] > 0]  # Eliminar cantidades negativas o cero

print(f"✅ Después de limpieza: {df.shape}")
print(f"Tipo corregido: {df['cantidadComprada'].dtype}")
print(f"Rango: {df['cantidadComprada'].min()} a {df['cantidadComprada'].max()}")

# CORRECCIÓN 2: Verificar fechas
print("\n📅 Verificando fechas...")
print("Formato de fechas (muestra):", df['Fechacompra'].unique()[:5])

# CORRECCIÓN 3: Limpiar IDs
print("\n🆔 Limpiando IDs...")
df['IDCliente'] = df['IDCliente'].fillna(0).astype(int)
df['SKU'] = df['SKU'].fillna(0).astype(int)

# Guardar datos limpios
archivo_limpio = "zfacturacion_bogota_limpio.csv"
df.to_csv(archivo_limpio, index=False)
print(f"💾 Datos limpios guardados en: {archivo_limpio}")

print("\n🚀 EJECUTANDO ANÁLISIS CON DATOS CORREGIDOS...")
print("="*50)

try:
    # Ejecutar análisis con datos limpios
    results = comprehensive_forecast_enhanced(archivo_limpio)
    
    if results and results['best_predictions'] is not None:
        best_df = results['best_predictions']
        print(f"\n✅ ¡ANÁLISIS EXITOSO!")
        print(f"📊 Predicciones generadas: {len(best_df)}")
        print(f"💰 Proyección total Q4: {best_df['cantidad_sugerida'].sum():,.2f}")
        
        # Mostrar top 10 clientes
        if 'NombreCliente' in best_df.columns:
            top_clientes = best_df.groupby(['IDCliente', 'NombreCliente'])['cantidad_sugerida'].sum().sort_values(ascending=False).head(10)
            print(f"\n🏆 TOP 10 CLIENTES - PREDICCIÓN Q4:")
            for (id_cliente, nombre), total in top_clientes.items():
                print(f"   • {nombre}: {total:,.2f}")
        
        # Mostrar distribución por mes
        print(f"\n📅 DISTRIBUCIÓN Q4 POR MES:")
        monthly = best_df.groupby('Mes_prediccion')['cantidad_sugerida'].sum()
        meses = {10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
        for mes, total in monthly.items():
            print(f"   • {meses[mes]}: {total:,.2f}")
            
    else:
        print("❌ No se pudieron generar predicciones válidas")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    print("\n📋 Información de diagnóstico:")
    print(f"   • Registros después de limpieza: {len(df)}")
    print(f"   • Clientes únicos: {df['IDCliente'].nunique()}")
    print(f"   • SKUs únicos: {df['SKU'].nunique()}")
    print(f"   • Rango de fechas: {df['Fechacompra'].min()} a {df['Fechacompra'].max()}")

print("\n✅ Proceso completado")