# forecasting_simple_working.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def simple_forecasting_system(file_path):
    """
    Sistema de forecasting simplificado pero funcional
    """
    print("🚀 SISTEMA SIMPLIFICADO DE FORECASTING")
    print("="*50)
    
    # 1. Cargar datos
    df = pd.read_csv(file_path)
    print(f"📊 Datos originales: {df.shape}")
    
    # 2. Limpiar datos
    df['cantidadComprada'] = df['cantidadComprada'].astype(str).str.replace(',', '.')
    df['cantidadComprada'] = pd.to_numeric(df['cantidadComprada'], errors='coerce')
    df = df.dropna(subset=['cantidadComprada'])
    df = df[df['cantidadComprada'] > 0]
    
    # Convertir fecha
    df['Fechacompra'] = pd.to_datetime(df['Fechacompra'], format='%m/%d/%Y', errors='coerce')
    df = df.dropna(subset=['Fechacompra'])
    
    # Limpiar IDs
    df['IDCliente'] = df['IDCliente'].fillna(0).astype(int)
    df['SKU'] = df['SKU'].fillna(0).astype(int)
    
    print(f"📊 Datos limpios: {df.shape}")
    
    # 3. Generar predicciones usando promedio móvil simple
    predicciones = []
    
    for (cliente, sku), grupo in df.groupby(['IDCliente', 'SKU']):
        if len(grupo) < 2:
            continue
            
        grupo_ordenado = grupo.sort_values('Fechacompra')
        
        # Calcular promedio de últimas 3 compras (o todas si son menos)
        ultimas_compras = grupo_ordenado['cantidadComprada'].tail(3)
        promedio_base = ultimas_compras.mean()
        
        # Factor estacional para Q4 (15% incremento)
        factor_estacional = 1.15
        
        # Obtener info del cliente
        info_cliente = grupo_ordenado.iloc[-1]
        
        # Generar predicciones para Oct, Nov, Dic
        for mes in [10, 11, 12]:
            prediccion = {
                'IDCliente': cliente,
                'NombreCliente': info_cliente.get('NombreCliente', ''),
                'SKU': sku,
                'Descripción': info_cliente.get('Descripción de SKU', ''),
                'Grupo_Material_1': info_cliente.get('Grupo de materiales 1', ''),
                'Grupo_Vendedores': info_cliente.get('Grupo de vendedores', ''),
                'zona_ventas': info_cliente.get('zona de ventas', 0),
                'Mes_prediccion': mes,
                'Trimestre': 4,
                'Año': 2025,
                'cantidad_sugerida': round(promedio_base * factor_estacional, 2),
                'registros_historicos': len(grupo_ordenado),
                'ultima_compra': grupo_ordenado['Fechacompra'].max().strftime('%Y-%m-%d'),
                'promedio_historico': round(grupo_ordenado['cantidadComprada'].mean(), 2)
            }
            predicciones.append(prediccion)
    
    # 4. Crear DataFrame de resultados
    df_predicciones = pd.DataFrame(predicciones)
    
    # 5. Generar reportes
    print(f"\n✅ PREDICCIONES GENERADAS")
    print(f"📊 Total predicciones: {len(df_predicciones)}")
    print(f"💰 Proyección total Q4: {df_predicciones['cantidad_sugerida'].sum():,.2f}")
    
    # Top 10 clientes por proyección
    if 'NombreCliente' in df_predicciones.columns:
        top_clientes = df_predicciones.groupby(['IDCliente', 'NombreCliente'])['cantidad_sugerida'].sum().sort_values(ascending=False).head(10)
        print(f"\n🏆 TOP 10 CLIENTES - PROYECCIÓN Q4:")
        for (id_cliente, nombre), total in top_clientes.items():
            print(f"   • {nombre} (ID: {id_cliente}): {total:,.2f}")
    
    # Distribución mensual
    print(f"\n📅 DISTRIBUCIÓN Q4 POR MES:")
    monthly_dist = df_predicciones.groupby('Mes_prediccion')['cantidad_sugerida'].sum()
    meses = {10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    for mes, total in monthly_dist.items():
        print(f"   • {meses[mes]}: {total:,.2f}")
    
    # Top grupos de materiales
    if 'Grupo_Material_1' in df_predicciones.columns:
        print(f"\n🏗️ TOP GRUPOS DE MATERIALES:")
        top_grupos = df_predicciones.groupby('Grupo_Material_1')['cantidad_sugerida'].sum().sort_values(ascending=False).head(5)
        for grupo, total in top_grupos.items():
            print(f"   • {grupo}: {total:,.2f}")
    
    # 6. Exportar resultados
    archivo_salida = 'predicciones_q4_simple.csv'
    df_predicciones.to_csv(archivo_salida, index=False, encoding='utf-8-sig')
    print(f"\n✅ Resultados exportados a: {archivo_salida}")
    
    # 7. Crear reporte por cliente
    reporte_clientes = df_predicciones.groupby(['IDCliente', 'NombreCliente']).agg({
        'cantidad_sugerida': 'sum',
        'SKU': 'count',
        'Grupo_Material_1': lambda x: ', '.join(x.unique()),
        'zona_ventas': 'first',
        'Grupo_Vendedores': 'first'
    }).round(2)
    
    reporte_clientes.columns = ['Total_Q4', 'Num_SKUs', 'Grupos_Materiales', 'Zona', 'Vendedor']
    reporte_clientes = reporte_clientes.sort_values('Total_Q4', ascending=False)
    
    archivo_clientes = 'reporte_clientes_q4.csv'
    reporte_clientes.to_csv(archivo_clientes, encoding='utf-8-sig')
    print(f"✅ Reporte por clientes exportado a: {archivo_clientes}")
    
    return df_predicciones, reporte_clientes

# Ejecutar análisis
if __name__ == "__main__":
    archivo = "zfacturacion_bogota_limpio.csv"
    
    try:
        predicciones, reporte_clientes = simple_forecasting_system(archivo)
        print("\n🎉 ¡ANÁLISIS COMPLETADO CON ÉXITO!")
        
        # Mostrar muestra de predicciones
        print(f"\n📋 MUESTRA DE PREDICCIONES:")
        print(predicciones[['IDCliente', 'NombreCliente', 'SKU', 'Mes_prediccion', 'cantidad_sugerida']].head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Verifica que existe el archivo zfacturacion_bogota_limpio.csv")