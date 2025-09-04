#!/usr/bin/env python3
"""
Sistema de Forecasting para Cliente Específico
Análisis detallado y predicciones para un solo cliente
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class ForecastClienteEspecifico:
    """Sistema para analizar y predecir un cliente específico"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = None
        self.cliente_seleccionado = None
        self.datos_cliente = None
        self.predicciones_cliente = []
        
    def cargar_datos(self):
        """Cargar y preparar datos"""
        print("🔄 Cargando datos...")
        
        self.df = pd.read_csv(self.csv_file_path, delimiter=';', encoding='utf-8')
        self.df.columns = self.df.columns.str.strip()
        
        # Limpiar datos
        self.df['cantidadComprada'] = (
            self.df['cantidadComprada']
            .astype(str)
            .str.replace('.', '', regex=False)  # Elimina puntos de miles
            .str.replace(',', '.', regex=False) # Convierte coma decimal a punto
            .astype(float)
        )
        self.df = self.df[self.df['cantidadComprada'] > 0].copy()
        
        # Crear fecha completa
        self.df['fecha_completa'] = pd.to_datetime(
            self.df['Año'].astype(str) + '-' + 
            self.df['Mes'].astype(str).str.zfill(2) + '-' + 
            self.df['Día'].astype(str).str.zfill(2)
        )
        
        print(f"✅ Datos cargados: {len(self.df):,} registros")
        return self
    
    def mostrar_clientes_disponibles(self, top=100):
        """Mostrar lista de clientes para seleccionar"""
        print(f"\n👥 CLIENTES DISPONIBLES (Top {top} por volumen):")
        print("="*70)
        
        # Calcular volumen total por cliente
        resumen_clientes = self.df.groupby('cliente').agg({
            'cantidadComprada': ['sum', 'count'],
            'Material': 'nunique',
            'Mes': 'nunique'
        }).round(2)
        
        resumen_clientes.columns = ['total_m2', 'transacciones', 'materiales_diferentes', 'meses_con_datos']
        resumen_clientes = resumen_clientes.sort_values('total_m2', ascending=False)
        
        print(f"{'#':<3} | {'Cliente':<35} | {'Total M2':<10} | {'Materiales':<10} | {'Meses':<6} | {'Trans.'}")
        print("-" * 70)
        
        for i, (cliente, data) in enumerate(resumen_clientes.head(top).iterrows(), 1):
            cliente_display = cliente[:33] + ".." if len(cliente) > 33 else cliente
            print(f"{i:<3} | {cliente_display:<35} | {data['total_m2']:<10,.0f} | "
                  f"{data['materiales_diferentes']:<10} | {data['meses_con_datos']:<6} | {data['transacciones']}")
        
        return resumen_clientes
    
    def seleccionar_cliente(self, nombre_cliente=None):
        """Seleccionar cliente específico para análisis"""
        
        if nombre_cliente is None:
            # Mostrar clientes disponibles
            resumen = self.mostrar_clientes_disponibles()
            
            print(f"\n🎯 SELECCIONAR CLIENTE:")
            print("Opciones:")
            print("1. Escribir número de la lista de arriba")
            print("2. Escribir nombre completo o parte del nombre")
            
            seleccion = input("\nTu selección: ").strip()
            
            # Si es un número
            if seleccion.isdigit():
                num = int(seleccion)
                if 1 <= num <= len(resumen):
                    nombre_cliente = resumen.index[num-1]
                else:
                    print("❌ Número fuera de rango")
                    return None
            
            # Si es texto, buscar coincidencias
            else:
                coincidencias = [cliente for cliente in self.df['cliente'].unique() 
                               if seleccion.lower() in cliente.lower()]
                
                if len(coincidencias) == 0:
                    print(f"❌ No se encontró ningún cliente con '{seleccion}'")
                    return None
                elif len(coincidencias) == 1:
                    nombre_cliente = coincidencias[0]
                else:
                    print(f"\n🔍 Se encontraron {len(coincidencias)} coincidencias:")
                    for i, cliente in enumerate(coincidencias[:10], 1):
                        print(f"{i}. {cliente}")
                    
                    try:
                        num = int(input("\nSelecciona el número: "))
                        if 1 <= num <= len(coincidencias):
                            nombre_cliente = coincidencias[num-1]
                        else:
                            print("❌ Número inválido")
                            return None
                    except:
                        print("❌ Entrada inválida")
                        return None
        
        # Filtrar datos del cliente seleccionado
        self.cliente_seleccionado = nombre_cliente
        self.datos_cliente = self.df[self.df['cliente'] == nombre_cliente].copy()
        
        print(f"\n✅ Cliente seleccionado: {nombre_cliente}")
        print(f"📊 Registros encontrados: {len(self.datos_cliente)}")
        
        return self
    
    def analizar_cliente(self):
        """Análisis completo del cliente seleccionado"""
        
        if self.datos_cliente is None:
            print("❌ Primero debes seleccionar un cliente")
            return None
        
        print(f"\n📊 ANÁLISIS DETALLADO: {self.cliente_seleccionado}")
        print("="*70)
        
        # Métricas generales
        total_m2 = self.datos_cliente['cantidadComprada'].sum()
        total_transacciones = len(self.datos_cliente)
        materiales_diferentes = self.datos_cliente['Material'].nunique()
        meses_con_datos = self.datos_cliente['Mes'].nunique()
        promedio_mensual = total_m2 / meses_con_datos
        
        print(f"💰 MÉTRICAS GENERALES:")
        print(f"   • Total M2 comprados: {total_m2:,.2f}")
        print(f"   • Total transacciones: {total_transacciones:,}")
        print(f"   • Materiales diferentes: {materiales_diferentes}")
        print(f"   • Meses con compras: {meses_con_datos}")
        print(f"   • Promedio mensual: {promedio_mensual:,.2f} M2")
        
        # Análisis por material
        print(f"\n🔧 TOP 10 MATERIALES DEL CLIENTE:")
        materiales_resumen = self.datos_cliente.groupby('Material').agg({
            'cantidadComprada': ['sum', 'count', 'mean'],
            'Mes': 'nunique'
        }).round(2)
        
        materiales_resumen.columns = ['total_m2', 'transacciones', 'promedio_compra', 'meses_activo']
        materiales_resumen = materiales_resumen.sort_values('total_m2', ascending=False)
        
        print(f"{'Material':<30} | {'Total M2':<10} | {'Trans.':<6} | {'Meses':<6} | {'Prom/Trans'}")
        print("-" * 70)
        
        for material, data in materiales_resumen.head(10).iterrows():
            material_display = material[:28] + ".." if len(material) > 28 else material
            print(f"{material_display:<30} | {data['total_m2']:<10,.0f} | "
                  f"{data['transacciones']:<6} | {data['meses_activo']:<6} | {data['promedio_compra']:<.1f}")
        
        # Análisis temporal
        print(f"\n📅 EVOLUCIÓN MENSUAL:")
        evolucion_mensual = self.datos_cliente.groupby(['Año', 'Mes']).agg({
            'cantidadComprada': 'sum',
            'Material': 'nunique'
        }).round(2)
        evolucion_mensual.columns = ['total_m2', 'materiales_diferentes']
        
        print(f"{'Año-Mes':<10} | {'M2 Total':<10} | {'Materiales'}")
        print("-" * 35)
        for (año, mes), data in evolucion_mensual.iterrows():
            print(f"{año}-{mes:02d}      | {data['total_m2']:<10,.0f} | {data['materiales_diferentes']}")
        
        return materiales_resumen
    
    def predecir_cliente_completo(self, meses_forecast=4):
        """Generar predicciones para todos los materiales del cliente"""
        
        if self.datos_cliente is None:
            print("❌ Primero debes seleccionar un cliente")
            return None
        
        print(f"\n🔮 GENERANDO PREDICCIONES PARA: {self.cliente_seleccionado}")
        print(f"   Próximos {meses_forecast} meses")
        print("="*70)
        
        # Preparar datos por cliente-material-mes
        datos_agrupados = self.datos_cliente.groupby(['Material', 'Año', 'Mes']).agg({
            'cantidadComprada': 'sum',
            'fecha_completa': 'max'
        }).reset_index()
        
        datos_agrupados['fecha_mes'] = pd.to_datetime(
            datos_agrupados['Año'].astype(str) + '-' + 
            datos_agrupados['Mes'].astype(str).str.zfill(2) + '-01'
        )
        
        self.predicciones_cliente = []
        
        # Generar predicciones para cada material
        for material in datos_agrupados['Material'].unique():
            datos_material = datos_agrupados[
                datos_agrupados['Material'] == material
            ].copy().sort_values('fecha_mes')
            
            if len(datos_material) >= 3:  # Mínimo 3 meses para predecir
                prediccion = self.predecir_material_cliente(material, datos_material, meses_forecast)
                if prediccion:
                    self.predicciones_cliente.append(prediccion)
        
        # Mostrar resumen de predicciones
        self.mostrar_resumen_predicciones()
        
        return self.predicciones_cliente
    
    def predecir_material_cliente(self, material, datos_material, meses_forecast=4):
        """Predecir un material específico del cliente"""
        
        # Preparar variables
        datos_material['periodo'] = range(1, len(datos_material) + 1)
        datos_material['mes_numero'] = datos_material['Mes']
        
        # Media móvil si hay suficientes datos
        if len(datos_material) >= 4:
            datos_material['media_movil'] = datos_material['cantidadComprada'].rolling(
                window=3, min_periods=1
            ).mean()
            features = ['periodo', 'mes_numero', 'media_movil']
        else:
            features = ['periodo', 'mes_numero']
        
        # Preparar modelo
        X = datos_material[features].values
        y = datos_material['cantidadComprada'].values
        
        # Seleccionar modelo
        if len(datos_material) >= 6:
            modelo = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)
        else:
            modelo = LinearRegression()
        
        modelo.fit(X, y)
        
        # Calcular precisión
        y_pred = modelo.predict(X)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.all(y > 0) else 100
        
        # Generar predicciones futuras
        predicciones_futuras = []
        ultimo_periodo = datos_material['periodo'].max()
        ultimo_mes = datos_material['Mes'].max()
        
        for i in range(1, meses_forecast + 1):
            nuevo_periodo = ultimo_periodo + i
            nuevo_mes = ((ultimo_mes + i - 1) % 12) + 1
            
            if len(datos_material) >= 4:
                nueva_media_movil = datos_material['cantidadComprada'].tail(3).mean()
                X_nuevo = np.array([[nuevo_periodo, nuevo_mes, nueva_media_movil]])
            else:
                X_nuevo = np.array([[nuevo_periodo, nuevo_mes]])
            
            prediccion = max(0, modelo.predict(X_nuevo)[0])
            
            predicciones_futuras.append({
                'mes': nuevo_mes,
                'mes_nombre': self.obtener_nombre_mes(nuevo_mes),
                'prediccion': round(prediccion, 2)
            })
        
        return {
            'material': material,
            'datos_historicos': datos_material,
            'predicciones': predicciones_futuras,
            'mape': mape,
            'total_historico': datos_material['cantidadComprada'].sum(),
            'total_predicho': sum([p['prediccion'] for p in predicciones_futuras]),
            'meses_datos': len(datos_material)
        }
    
    def obtener_nombre_mes(self, mes_num):
        """Convertir número de mes a nombre"""
        meses = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        return meses.get(mes_num, f'Mes_{mes_num}')
    
    def mostrar_resumen_predicciones(self):
        """Mostrar resumen de todas las predicciones del cliente"""
        
        if not self.predicciones_cliente:
            print("❌ No hay predicciones disponibles")
            return
        
        print(f"\n📈 RESUMEN DE PREDICCIONES:")
        print("="*80)
        
        total_historico = sum([p['total_historico'] for p in self.predicciones_cliente])
        total_predicho = sum([p['total_predicho'] for p in self.predicciones_cliente])
        
        print(f"📊 TOTALES DEL CLIENTE:")
        print(f"   • M2 histórico total: {total_historico:,.2f}")
        print(f"   • M2 predicho (4 meses): {total_predicho:,.2f}")
        print(f"   • Cambio vs promedio histórico: {((total_predicho/4) / (total_historico/8) - 1) * 100:+.1f}%")
        print(f"   • Materiales predictibles: {len(self.predicciones_cliente)}")
        
        print(f"\n🔧 PREDICCIONES POR MATERIAL:")
        print(f"{'Material':<35} | {'MAPE':<6} | {'Sep':<8} | {'Oct':<8} | {'Nov':<8} | {'Dic':<8} | {'Total'}")
        print("-" * 80)
        
        # Ordenar por total predicho
        predicciones_ordenadas = sorted(
            self.predicciones_cliente, 
            key=lambda x: x['total_predicho'], 
            reverse=True
        )
        
        totales_mensuales = [0, 0, 0, 0]
        
        for pred in predicciones_ordenadas:
            material_display = pred['material'][:33] + ".." if len(pred['material']) > 33 else pred['material']
            mape = pred['mape']
            
            # Color basado en MAPE
            if mape < 25:
                mape_display = f"{mape:.0f}% ✅"
            elif mape < 50:
                mape_display = f"{mape:.0f}% ⚠️"
            else:
                mape_display = f"{mape:.0f}% ❌"
            
            preds = pred['predicciones']
            pred_sep = preds[0]['prediccion'] if len(preds) > 0 else 0
            pred_oct = preds[1]['prediccion'] if len(preds) > 1 else 0
            pred_nov = preds[2]['prediccion'] if len(preds) > 2 else 0
            pred_dic = preds[3]['prediccion'] if len(preds) > 3 else 0
            
            # Sumar a totales
            totales_mensuales[0] += pred_sep
            totales_mensuales[1] += pred_oct
            totales_mensuales[2] += pred_nov
            totales_mensuales[3] += pred_dic
            
            print(f"{material_display:<35} | {mape_display:<6} | {pred_sep:<8,.0f} | "
                  f"{pred_oct:<8,.0f} | {pred_nov:<8,.0f} | {pred_dic:<8,.0f} | {pred['total_predicho']:<8,.0f}")
        
        # Fila de totales
        print("-" * 80)
        print(f"{'TOTALES':<35} | {'':6} | {totales_mensuales[0]:<8,.0f} | "
              f"{totales_mensuales[1]:<8,.0f} | {totales_mensuales[2]:<8,.0f} | "
              f"{totales_mensuales[3]:<8,.0f} | {total_predicho:<8,.0f}")
        
        print(f"\n💡 INTERPRETACIÓN:")
        print(f"   ✅ MAPE <25%: Predicción muy confiable")
        print(f"   ⚠️ MAPE 25-50%: Predicción útil con precaución")  
        print(f"   ❌ MAPE >50%: Revisar datos, muy impredecible")
    
    def generar_grafico_cliente(self, archivo_grafico=None):
        """Generar gráfico específico para el cliente"""
        
        if not self.predicciones_cliente:
            print("❌ No hay predicciones para graficar")
            return
        
        if archivo_grafico is None:
            nombre_cliente_clean = "".join(c for c in self.cliente_seleccionado if c.isalnum() or c in (' ', '-', '_')).rstrip()
            archivo_grafico = f'predicciones_{nombre_cliente_clean[:30]}.png'
        
        # Tomar los top 6 materiales
        top_materiales = sorted(self.predicciones_cliente, key=lambda x: x['total_predicho'], reverse=True)[:6]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle(f'Predicciones para: {self.cliente_seleccionado[:50]}...', fontsize=14, fontweight='bold')
        
        # Gráfico 1: Evolución por material
        ax1.set_title('Evolución Histórica y Predicciones por Material', fontsize=12)
        
        colores = plt.cm.Set3(np.linspace(0, 1, len(top_materiales)))
        
        for i, pred in enumerate(top_materiales):
            datos_hist = pred['datos_historicos']
            
            # Datos históricos
            valores_hist = datos_hist['cantidadComprada'].values
            meses_hist = list(range(len(valores_hist)))
            
            # Predicciones
            valores_pred = [p['prediccion'] for p in pred['predicciones']]
            meses_pred = list(range(len(valores_hist), len(valores_hist) + len(valores_pred)))
            
            # Plot histórico
            ax1.plot(meses_hist, valores_hist, 'o-', color=colores[i], 
                    linewidth=2, markersize=6, label=f"{pred['material'][:20]}...")
            
            # Plot predicciones
            ax1.plot([meses_hist[-1]] + meses_pred, [valores_hist[-1]] + valores_pred, 
                    '--', color=colores[i], linewidth=2, alpha=0.7)
        
        ax1.axvline(x=len(top_materiales[0]['datos_historicos'])-0.5, 
                   color='red', linestyle=':', alpha=0.7, label='Inicio Predicciones')
        ax1.set_xlabel('Período')
        ax1.set_ylabel('M2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Totales mensuales
        ax2.set_title('Predicciones Totales por Mes', fontsize=12)
        
        meses_nombres = ['Sep 2025', 'Oct 2025', 'Nov 2025', 'Dic 2025']
        totales_meses = [0, 0, 0, 0]
        
        for pred in self.predicciones_cliente:
            for i, pred_mes in enumerate(pred['predicciones']):
                if i < 4:
                    totales_meses[i] += pred_mes['prediccion']
        
        barras = ax2.bar(meses_nombres, totales_meses, color='lightcoral', alpha=0.8)
        
        # Agregar valores en las barras
        for barra, valor in zip(barras, totales_meses):
            ax2.text(barra.get_x() + barra.get_width()/2, barra.get_height() + valor*0.01,
                    f'{valor:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('M2 Totales')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfico guardado: {archivo_grafico}")
        return archivo_grafico
    
    def exportar_excel_cliente(self, archivo_excel=None):
        """Exportar predicciones del cliente a Excel"""
        
        if not self.predicciones_cliente:
            print("❌ No hay predicciones para exportar")
            return
        
        if archivo_excel is None:
            nombre_cliente_clean = "".join(c for c in self.cliente_seleccionado if c.isalnum() or c in (' ', '-', '_')).rstrip()
            archivo_excel = f'predicciones_{nombre_cliente_clean[:30]}.xlsx'
        
        datos_excel = []
        
        for pred in self.predicciones_cliente:
            fila = {
                'Cliente': self.cliente_seleccionado,
                'Material': pred['material'],
                'Total_Historico_M2': pred['total_historico'],
                'Meses_Datos': pred['meses_datos'],
                'MAPE_Precision': pred['mape'],
                'Septiembre_2025': pred['predicciones'][0]['prediccion'] if len(pred['predicciones']) > 0 else 0,
                'Octubre_2025': pred['predicciones'][1]['prediccion'] if len(pred['predicciones']) > 1 else 0,
                'Noviembre_2025': pred['predicciones'][2]['prediccion'] if len(pred['predicciones']) > 2 else 0,
                'Diciembre_2025': pred['predicciones'][3]['prediccion'] if len(pred['predicciones']) > 3 else 0,
                'Total_Predicho_4_Meses': pred['total_predicho']
            }
            datos_excel.append(fila)
        
        df_excel = pd.DataFrame(datos_excel)
        df_excel.to_excel(archivo_excel, index=False)
        
        print(f"✅ Excel generado: {archivo_excel}")
        return archivo_excel


def main_cliente_especifico():
    """Función principal para analizar cliente específico"""
    
    print("🎯 ANÁLISIS Y PREDICCIÓN DE CLIENTE ESPECÍFICO")
    print("="*60)
    
    # Solicitar archivo
    archivo_csv = input("📁 Ingresa la ruta del archivo CSV: ").strip()
    if not archivo_csv:
        archivo_csv = "datos_ventas.csv"
        print(f"   Usando archivo por defecto: {archivo_csv}")
    
    try:
        # Inicializar sistema
        sistema = ForecastClienteEspecifico(archivo_csv)
        sistema.cargar_datos()
        
        # Seleccionar cliente
        sistema.seleccionar_cliente()
        
        if sistema.cliente_seleccionado:
            # Análisis completo
            sistema.analizar_cliente()
            
            # Generar predicciones
            predicciones = sistema.predecir_cliente_completo(meses_forecast=4)
            
            if predicciones:
                # Generar reportes
                archivo_grafico = sistema.generar_grafico_cliente()
                archivo_excel = sistema.exportar_excel_cliente()
                
                print(f"\n🎉 ¡ANÁLISIS COMPLETADO!")
                print(f"📊 Archivos generados:")
                print(f"   • {archivo_grafico} (gráficos)")
                print(f"   • {archivo_excel} (datos Excel)")
            else:
                print("❌ No se pudieron generar predicciones para este cliente")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_cliente_especifico()