#!/usr/bin/env python3
"""
Sistema de Forecasting Granular por Cliente-Material-Cantidad
Predicción específica a nivel de SKU/Material por cliente
Optimizado para planificación de inventarios y compras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class ForecastClienteMaterial:
    """Sistema especializado para forecast por cliente-material"""
    
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.df = None
        self.df_cliente_material = None
        self.combinaciones_validas = []
        self.predicciones_detalladas = []
        
    def cargar_y_preparar_datos(self):
        """Cargar datos optimizado para análisis cliente-material"""
        print("🔄 Cargando datos para análisis cliente-material...")
        
        # Cargar datos
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

        
        # Crear fecha completa
        self.df['fecha_completa'] = pd.to_datetime(
            self.df['Año'].astype(str) + '-' + 
            self.df['Mes'].astype(str).str.zfill(2) + '-' + 
            self.df['Día'].astype(str).str.zfill(2)
        )
        
        # Filtrar datos válidos
        self.df = self.df[self.df['cantidadComprada'] > 0].copy()
        
        # Crear agregación por cliente-material-mes
        self.df_cliente_material = self.df.groupby([
            'cliente', 'Material', 'Año', 'Mes'
        ]).agg({
            'cantidadComprada': ['sum', 'count', 'mean'],
            'sku': 'nunique',
            'fecha_completa': 'max'
        }).reset_index()
        
        # Aplanar columnas multinivel
        self.df_cliente_material.columns = [
            'cliente', 'material', 'año', 'mes',
            'cantidad_total', 'transacciones', 'cantidad_promedio',
            'skus_diferentes', 'ultima_fecha'
        ]
        
        # Crear fecha mensual
        self.df_cliente_material['fecha_mes'] = pd.to_datetime(
            self.df_cliente_material['año'].astype(str) + '-' + 
            self.df_cliente_material['mes'].astype(str).str.zfill(2) + '-01'
        )
        
        print(f"✅ Datos preparados: {len(self.df):,} registros originales")
        print(f"📊 Combinaciones cliente-material-mes: {len(self.df_cliente_material):,}")
        
        return self
    
    def identificar_combinaciones_predictibles(self, min_meses=3, min_cantidad_total=100):
        """Identificar qué combinaciones cliente-material son predictibles"""
        
        print(f"\n🔍 Identificando combinaciones predictibles...")
        print(f"   Criterios: ≥{min_meses} meses, ≥{min_cantidad_total} M2 total")
        
        # Analizar cada combinación cliente-material
        combinacion_stats = self.df_cliente_material.groupby(['cliente', 'material']).agg({
            'cantidad_total': ['sum', 'mean', 'std', 'count'],
            'mes': ['nunique', 'min', 'max'],
            'transacciones': 'sum',
            'ultima_fecha': 'max'
        }).round(2)
        
        # Aplanar columnas
        combinacion_stats.columns = ['_'.join(col) for col in combinacion_stats.columns]
        
        # Aplicar filtros de calidad
        combinaciones_validas = combinacion_stats[
            (combinacion_stats['cantidad_total_count'] >= min_meses) &
            (combinacion_stats['mes_nunique'] >= min_meses) &
            (combinacion_stats['cantidad_total_sum'] >= min_cantidad_total)
        ].copy()
        
        # Calcular métricas adicionales
        combinaciones_validas['coef_variacion'] = (
            combinaciones_validas['cantidad_total_std'] / 
            combinaciones_validas['cantidad_total_mean']
        ).fillna(0)
        
        combinaciones_validas['regularidad'] = (
            combinaciones_validas['cantidad_total_count'] / combinaciones_validas['mes_nunique']
        )
        
        # Ordenar por importancia (volumen + estabilidad)
        combinaciones_validas['score_importancia'] = (
            combinaciones_validas['cantidad_total_sum'] * 
            (1 - np.minimum(combinaciones_validas['coef_variacion'], 1)) *
            np.minimum(combinaciones_validas['regularidad'], 2)
        )
        
        self.combinaciones_validas = combinaciones_validas.sort_values(
            'score_importancia', ascending=False
        )
        
        print(f"✅ Combinaciones válidas encontradas: {len(self.combinaciones_validas):,}")
        
        # Mostrar top 10 combinaciones
        print(f"\n📊 TOP 10 COMBINACIONES MÁS IMPORTANTES:")
        for i, ((cliente, material), data) in enumerate(self.combinaciones_validas.head(10).iterrows(), 1):
            cliente_short = cliente[:25] + "..." if len(cliente) > 25 else cliente
            material_short = material[:30] + "..." if len(material) > 30 else material
            total_cantidad = data['cantidad_total_sum']
            meses_datos = data['mes_nunique']
            cv = data['coef_variacion']
            
            print(f"{i:2d}. {cliente_short:28} | {material_short:33} | {total_cantidad:8,.0f} M2 | {meses_datos} meses | CV:{cv:.2f}")
        
        return self
    
    def generar_forecast_combinacion(self, cliente, material, meses_forecast=4):
        """Generar forecast para una combinación específica cliente-material"""
        
        # Obtener datos históricos de la combinación
        datos_combinacion = self.df_cliente_material[
            (self.df_cliente_material['cliente'] == cliente) &
            (self.df_cliente_material['material'] == material)
        ].copy().sort_values('fecha_mes')
        
        if len(datos_combinacion) < 3:
            return None
        
        # Preparar variables predictoras
        datos_combinacion['periodo'] = range(1, len(datos_combinacion) + 1)
        datos_combinacion['mes_numero'] = datos_combinacion['mes']
        
        # Variables adicionales si hay suficientes datos
        if len(datos_combinacion) >= 4:
            datos_combinacion['media_movil'] = datos_combinacion['cantidad_total'].rolling(
                window=3, min_periods=1
            ).mean()
        else:
            datos_combinacion['media_movil'] = datos_combinacion['cantidad_total']
        
        # Calcular índice estacional si hay datos de múltiples meses
        if datos_combinacion['mes'].nunique() >= 3:
            promedio_por_mes = datos_combinacion.groupby('mes')['cantidad_total'].mean()
            promedio_general = datos_combinacion['cantidad_total'].mean()
            indice_estacional = promedio_por_mes / promedio_general
            datos_combinacion['indice_estacional'] = datos_combinacion['mes'].map(indice_estacional).fillna(1.0)
        else:
            datos_combinacion['indice_estacional'] = 1.0
        
        # Preparar datos para modelo
        features = ['periodo', 'mes_numero']
        if len(datos_combinacion) >= 4:
            features.extend(['media_movil', 'indice_estacional'])
        
        X = datos_combinacion[features].fillna(method='ffill')
        y = datos_combinacion['cantidad_total'].values
        
        # Seleccionar y entrenar modelo
        if len(datos_combinacion) >= 6:
            # Random Forest para series más largas
            modelo = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42)
        else:
            # Regresión lineal para series cortas
            modelo = LinearRegression()
        
        modelo.fit(X, y)
        
        # Calcular precisión histórica
        y_pred = modelo.predict(X)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100 if np.all(y > 0) else 100
        
        # Generar predicciones futuras
        predicciones_futuras = []
        ultimo_periodo = datos_combinacion['periodo'].max()
        ultimo_mes = datos_combinacion['mes'].max()
        
        for i in range(1, meses_forecast + 1):
            nuevo_periodo = ultimo_periodo + i
            nuevo_mes = ((ultimo_mes + i - 1) % 12) + 1
            
            # Calcular nuevas features
            nuevas_features = {
                'periodo': nuevo_periodo,
                'mes_numero': nuevo_mes
            }
            
            if len(datos_combinacion) >= 4:
                # Media móvil estimada
                nuevas_features['media_movil'] = datos_combinacion['cantidad_total'].tail(3).mean()
                
                # Índice estacional
                if nuevo_mes in indice_estacional:
                    nuevas_features['indice_estacional'] = indice_estacional[nuevo_mes]
                else:
                    nuevas_features['indice_estacional'] = 1.0
            
            # Crear array de features
            X_nuevo = np.array([[nuevas_features[feat] for feat in features]])
            
            # Realizar predicción
            prediccion = modelo.predict(X_nuevo)[0]
            prediccion = max(0, prediccion)  # No predicciones negativas
            
            predicciones_futuras.append({
                'mes': nuevo_mes,
                'periodo': nuevo_periodo,
                'prediccion': round(prediccion, 2),
                'mes_nombre': self.obtener_nombre_mes(nuevo_mes)
            })
        
        return {
            'cliente': cliente,
            'material': material,
            'datos_historicos': datos_combinacion,
            'predicciones': predicciones_futuras,
            'mape': mape,
            'total_historico': datos_combinacion['cantidad_total'].sum(),
            'total_predicho': sum([p['prediccion'] for p in predicciones_futuras]),
            'meses_datos': len(datos_combinacion)
        }
    
    def obtener_nombre_mes(self, mes_num):
        """Convertir número de mes a nombre"""
        meses = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        return meses.get(mes_num, f'Mes_{mes_num}')
    
    def ejecutar_forecast_completo(self, top_combinaciones=100, meses_forecast=4):
        """Ejecutar forecast para las top combinaciones cliente-material"""
        
        print(f"\n🔮 INICIANDO FORECAST POR CLIENTE-MATERIAL")
        print(f"   Combinaciones a procesar: {top_combinaciones}")
        print(f"   Meses de predicción: {meses_forecast}")
        print("=" * 70)
        
        resultados = []
        combinaciones_procesadas = 0
        
        # Procesar top combinaciones
        for (cliente, material), data in self.combinaciones_validas.head(top_combinaciones).iterrows():
            combinaciones_procesadas += 1
            
            if combinaciones_procesadas % 10 == 0:
                print(f"   Procesado: {combinaciones_procesadas}/{top_combinaciones}")
            
            # Generar forecast para esta combinación
            resultado = self.generar_forecast_combinacion(cliente, material, meses_forecast)
            
            if resultado and resultado['mape'] < 100:  # Filtrar resultados muy imprecisos
                resultados.append(resultado)
        
        self.predicciones_detalladas = sorted(resultados, key=lambda x: x['total_predicho'], reverse=True)
        
        print(f"✅ Forecast completado: {len(self.predicciones_detalladas)} combinaciones exitosas")
        
        return self
    
    def generar_reporte_detallado(self, archivo_salida='forecast_cliente_material.html'):
        """Generar reporte detallado por cliente-material"""
        
        print(f"\n📊 Generando reporte detallado: {archivo_salida}")
        
        # Calcular totales y resúmenes
        total_predicho = sum([r['total_predicho'] for r in self.predicciones_detalladas])
        total_historico = sum([r['total_historico'] for r in self.predicciones_detalladas])
        
        # Agrupar por cliente para resumen
        resumen_clientes = {}
        for pred in self.predicciones_detalladas:
            cliente = pred['cliente']
            if cliente not in resumen_clientes:
                resumen_clientes[cliente] = {
                    'materiales_count': 0,
                    'total_historico': 0,
                    'total_predicho': 0,
                    'predicciones_mes': [0, 0, 0, 0]
                }
            
            resumen_clientes[cliente]['materiales_count'] += 1
            resumen_clientes[cliente]['total_historico'] += pred['total_historico']
            resumen_clientes[cliente]['total_predicho'] += pred['total_predicho']
            
            # Sumar predicciones por mes
            for i, pred_mes in enumerate(pred['predicciones']):
                if i < 4:
                    resumen_clientes[cliente]['predicciones_mes'][i] += pred_mes['prediccion']
        
        # Crear HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="es">
        <head>
            <meta charset="UTF-8">
            <title>Forecast Detallado: Cliente-Material-Cantidad</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 15px; 
                    background: #f5f6fa; 
                    font-size: 13px;
                }}
                .container {{ 
                    max-width: 1600px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 25px; 
                    border-radius: 10px; 
                    box-shadow: 0 3px 15px rgba(0,0,0,0.1); 
                }}
                h1 {{ 
                    color: #2c3e50; 
                    text-align: center; 
                    border-bottom: 3px solid #e74c3c; 
                    padding-bottom: 10px; 
                    font-size: 24px;
                }}
                h2 {{ 
                    color: #34495e; 
                    margin-top: 30px; 
                    border-left: 4px solid #e74c3c; 
                    padding-left: 10px; 
                    font-size: 18px;
                }}
                
                .resumen-cards {{ 
                    display: flex; 
                    justify-content: space-around; 
                    margin: 20px 0; 
                    flex-wrap: wrap; 
                }}
                .card {{ 
                    background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                    color: white; 
                    padding: 15px; 
                    border-radius: 8px; 
                    text-align: center; 
                    min-width: 160px; 
                    margin: 5px; 
                }}
                .card-valor {{ font-size: 20px; font-weight: bold; }}
                .card-label {{ font-size: 11px; opacity: 0.9; margin-top: 3px; }}
                
                .tabla-detallada {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 15px 0; 
                    font-size: 11px; 
                }}
                .tabla-detallada th {{ 
                    background: #2c3e50; 
                    color: white; 
                    padding: 8px 4px; 
                    text-align: center; 
                    font-size: 10px;
                }}
                .tabla-detallada td {{ 
                    padding: 6px 4px; 
                    border-bottom: 1px solid #ecf0f1; 
                    text-align: center; 
                }}
                .tabla-detallada tbody tr:hover {{ background-color: #f8f9fa; }}
                
                .cliente-col {{ text-align: left !important; max-width: 180px; font-weight: bold; }}
                .material-col {{ text-align: left !important; max-width: 200px; }}
                .cantidad-pred {{ font-weight: bold; color: #27ae60; }}
                .mape-excelente {{ color: #27ae60; font-weight: bold; }}
                .mape-bueno {{ color: #f39c12; font-weight: bold; }}
                .mape-revisar {{ color: #e74c3c; font-weight: bold; }}
                
                .seccion-resumen {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 20px 0; }}
                .totales-fila {{ background: #34495e; color: white; font-weight: bold; }}
                
                .filtros {{ background: #3498db; color: white; padding: 10px; border-radius: 5px; margin: 15px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎯 Forecast Detallado por Cliente-Material-Cantidad</h1>
                
                <div class="resumen-cards">
                    <div class="card">
                        <div class="card-valor">{len(self.predicciones_detalladas)}</div>
                        <div class="card-label">Combinaciones<br>Cliente-Material</div>
                    </div>
                    <div class="card">
                        <div class="card-valor">{len(resumen_clientes)}</div>
                        <div class="card-label">Clientes<br>Únicos</div>
                    </div>
                    <div class="card">
                        <div class="card-valor">{total_historico:,.0f}</div>
                        <div class="card-label">M2 Histórico<br>Total</div>
                    </div>
                    <div class="card">
                        <div class="card-valor">{total_predicho:,.0f}</div>
                        <div class="card-label">M2 Predicho<br>(4 meses)</div>
                    </div>
                    <div class="card">
                        <div class="card-valor">{((total_predicho/4) / (total_historico/8) - 1) * 100:+.1f}%</div>
                        <div class="card-label">Cambio vs<br>Promedio</div>
                    </div>
                </div>
                
                <div class="filtros">
                    📊 Showing top {len(self.predicciones_detalladas)} combinaciones cliente-material más importantes
                </div>
                
                <h2>📋 Predicciones Detalladas por Cliente-Material</h2>
                <table class="tabla-detallada">
                    <thead>
                        <tr>
                            <th>Cliente</th>
                            <th>Material</th>
                            <th>Hist.<br>Total</th>
                            <th>Hist.<br>Prom/Mes</th>
                            <th>MAPE</th>
                            <th>Sep<br>2025</th>
                            <th>Oct<br>2025</th>
                            <th>Nov<br>2025</th>
                            <th>Dic<br>2025</th>
                            <th>Total<br>Predicho</th>
                            <th>Meses<br>Datos</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Agregar filas de datos detallados
        for pred in self.predicciones_detalladas:
            cliente = pred['cliente'][:25] + "..." if len(pred['cliente']) > 25 else pred['cliente']
            material = pred['material'][:35] + "..." if len(pred['material']) > 35 else pred['material']
            
            hist_total = pred['total_historico']
            hist_prom = hist_total / pred['meses_datos']
            mape = pred['mape']
            total_pred = pred['total_predicho']
            meses_datos = pred['meses_datos']
            
            # Clases CSS para colores
            mape_class = 'mape-excelente' if mape < 25 else 'mape-bueno' if mape < 50 else 'mape-revisar'
            
            # Predicciones mensuales
            preds = pred['predicciones']
            pred_sep = preds[0]['prediccion'] if len(preds) > 0 else 0
            pred_oct = preds[1]['prediccion'] if len(preds) > 1 else 0
            pred_nov = preds[2]['prediccion'] if len(preds) > 2 else 0
            pred_dic = preds[3]['prediccion'] if len(preds) > 3 else 0
            
            html += f"""
                        <tr>
                            <td class="cliente-col">{cliente}</td>
                            <td class="material-col">{material}</td>
                            <td>{hist_total:,.0f}</td>
                            <td>{hist_prom:,.0f}</td>
                            <td class="{mape_class}">{mape:.0f}%</td>
                            <td class="cantidad-pred">{pred_sep:,.0f}</td>
                            <td class="cantidad-pred">{pred_oct:,.0f}</td>
                            <td class="cantidad-pred">{pred_nov:,.0f}</td>
                            <td class="cantidad-pred">{pred_dic:,.0f}</td>
                            <td><strong>{total_pred:,.0f}</strong></td>
                            <td>{meses_datos}</td>
                        </tr>
            """
        
        # Calcular totales finales
        totales_sep = sum([r['predicciones'][0]['prediccion'] for r in self.predicciones_detalladas if len(r['predicciones']) > 0])
        totales_oct = sum([r['predicciones'][1]['prediccion'] for r in self.predicciones_detalladas if len(r['predicciones']) > 1])
        totales_nov = sum([r['predicciones'][2]['prediccion'] for r in self.predicciones_detalladas if len(r['predicciones']) > 2])
        totales_dic = sum([r['predicciones'][3]['prediccion'] for r in self.predicciones_detalladas if len(r['predicciones']) > 3])
        
        html += f"""
                        <tr class="totales-fila">
                            <td colspan="2"><strong>TOTALES GENERALES</strong></td>
                            <td><strong>{total_historico:,.0f}</strong></td>
                            <td><strong>{total_historico/8:,.0f}</strong></td>
                            <td>-</td>
                            <td><strong>{totales_sep:,.0f}</strong></td>
                            <td><strong>{totales_oct:,.0f}</strong></td>
                            <td><strong>{totales_nov:,.0f}</strong></td>
                            <td><strong>{totales_dic:,.0f}</strong></td>
                            <td><strong>{total_predicho:,.0f}</strong></td>
                            <td>-</td>
                        </tr>
                    </tbody>
                </table>
                
                <h2>👥 Resumen Consolidado por Cliente</h2>
                <table class="tabla-detallada">
                    <thead>
                        <tr>
                            <th>Cliente</th>
                            <th>Materiales<br>Diferentes</th>
                            <th>Total<br>Histórico</th>
                            <th>Sep 2025</th>
                            <th>Oct 2025</th>
                            <th>Nov 2025</th>
                            <th>Dic 2025</th>
                            <th>Total<br>Predicho</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Agregar resumen por cliente
        for cliente, data in sorted(resumen_clientes.items(), key=lambda x: x[1]['total_predicho'], reverse=True):
            cliente_display = cliente[:40] + "..." if len(cliente) > 40 else cliente
            
            html += f"""
                        <tr>
                            <td class="cliente-col">{cliente_display}</td>
                            <td>{data['materiales_count']}</td>
                            <td>{data['total_historico']:,.0f}</td>
                            <td class="cantidad-pred">{data['predicciones_mes'][0]:,.0f}</td>
                            <td class="cantidad-pred">{data['predicciones_mes'][1]:,.0f}</td>
                            <td class="cantidad-pred">{data['predicciones_mes'][2]:,.0f}</td>
                            <td class="cantidad-pred">{data['predicciones_mes'][3]:,.0f}</td>
                            <td><strong>{data['total_predicho']:,.0f}</strong></td>
                        </tr>
            """
        
        html += f"""
                    </tbody>
                </table>
                
                <div class="seccion-resumen">
                    <h3>ℹ️ Información del Modelo</h3>
                    <ul>
                        <li><strong>Período histórico:</strong> Enero - Agosto 2025 (8 meses de datos)</li>
                        <li><strong>Predicción:</strong> Septiembre - Diciembre 2025 (4 meses adelante)</li>
                        <li><strong>Granularidad:</strong> Cliente + Material específico (máximo detalle)</li>
                        <li><strong>Filtros aplicados:</strong> ≥3 meses datos, ≥100 M2 total por combinación</li>
                        <li><strong>Modelos:</strong> Random Forest (series largas) + Regresión Lineal (series cortas)</li>
                        <li><strong>Variables:</strong> Tendencia, estacionalidad, media móvil, patrones históricos</li>
                        <li><strong>MAPE:</strong> &lt;25% Excelente, 25-50% Bueno, &gt;50% Revisar</li>
                    </ul>
                </div>
                
                <div style="margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 11px;">
                    <p>Reporte generado el {datetime.now().strftime("%d/%m/%Y a las %H:%M")}</p>
                    <p>Sistema de Forecasting Granular - Cliente-Material-Cantidad</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar archivo
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Reporte detallado guardado: {archivo_salida}")
        
        return archivo_salida
    
    def exportar_excel(self, archivo_excel='predicciones_cliente_material.xlsx'):
        """Exportar predicciones a Excel para análisis"""
        
        print(f"\n📤 Exportando a Excel: {archivo_excel}")
        
        # Preparar datos para Excel
        datos_excel = []
        
        for pred in self.predicciones_detalladas:
            fila_base = {
                'Cliente': pred['cliente'],
                'Material': pred['material'],
                'Total_Historico_M2': pred['total_historico'],
                'Promedio_Mensual_Historico': pred['total_historico'] / pred['meses_datos'],
                'Meses_con_Datos': pred['meses_datos'],
                'MAPE_Precision': pred['mape'],
                'Total_Predicho_4_Meses': pred['total_predicho']
            }
            
            # Agregar predicciones mensuales
            for i, pred_mes in enumerate(pred['predicciones']):
                fila_base[f'{pred_mes["mes_nombre"]}_2025'] = pred_mes['prediccion']
            
            datos_excel.append(fila_base)
        
        # Crear DataFrame y exportar
        df_excel = pd.DataFrame(datos_excel)
        df_excel.to_excel(archivo_excel, index=False)
        
        print(f"✅ Datos exportados a Excel: {archivo_excel}")
        print(f"   Registros: {len(datos_excel):,}")
        print(f"   Columnas: {len(df_excel.columns)}")
        
        return archivo_excel
    
    def mostrar_resumen_ejecutivo(self):
        """Mostrar resumen ejecutivo de las predicciones"""
        
        if not self.predicciones_detalladas:
            print("❌ No hay predicciones disponibles. Ejecuta forecast_completo primero.")
            return
        
        print("\n" + "="*80)
        print("📈 RESUMEN EJECUTIVO - FORECAST CLIENTE-MATERIAL")
        print("="*80)
        
        # Métricas generales
        total_combinaciones = len(self.predicciones_detalladas)
        total_historico = sum([r['total_historico'] for r in self.predicciones_detalladas])
        total_predicho = sum([r['total_predicho'] for r in self.predicciones_detalladas])
        
        # Clientes únicos
        clientes_unicos = len(set([r['cliente'] for r in self.predicciones_detalladas]))
        materiales_unicos = len(set([r['material'] for r in self.predicciones_detalladas]))
        
        # Precisión promedio
        mape_promedio = np.mean([r['mape'] for r in self.predicciones_detalladas])
        
        # Cambio vs histórico
        cambio_vs_historico = ((total_predicho/4) / (total_historico/8) - 1) * 100
        
        print(f"🎯 MÉTRICAS CLAVE:")
        print(f"   • Combinaciones cliente-material: {total_combinaciones:,}")
        print(f"   • Clientes únicos: {clientes_unicos:,}")
        print(f"   • Materiales únicos: {materiales_unicos:,}")
        print(f"   • M2 histórico total (8 meses): {total_historico:,.0f}")
        print(f"   • M2 predicho total (4 meses): {total_predicho:,.0f}")
        print(f"   • Promedio mensual histórico: {total_historico/8:,.0f} M2")
        print(f"   • Promedio mensual predicho: {total_predicho/4:,.0f} M2")
        print(f"   • Cambio vs promedio histórico: {cambio_vs_historico:+.1f}%")
        print(f"   • MAPE promedio del modelo: {mape_promedio:.1f}%")
        
        # Top 10 clientes por volumen predicho
        print(f"\n🏆 TOP 10 CLIENTES POR VOLUMEN PREDICHO:")
        clientes_volumen = {}
        for pred in self.predicciones_detalladas:
            cliente = pred['cliente']
            if cliente not in clientes_volumen:
                clientes_volumen[cliente] = 0
            clientes_volumen[cliente] += pred['total_predicho']
        
        top_clientes = sorted(clientes_volumen.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (cliente, volumen) in enumerate(top_clientes, 1):
            cliente_display = cliente[:40] + "..." if len(cliente) > 40 else cliente
            porcentaje = (volumen / total_predicho) * 100
            print(f"   {i:2d}. {cliente_display:43} | {volumen:8,.0f} M2 ({porcentaje:4.1f}%)")
        
        # Top 10 materiales por volumen predicho
        print(f"\n🔧 TOP 10 MATERIALES POR VOLUMEN PREDICHO:")
        materiales_volumen = {}
        for pred in self.predicciones_detalladas:
            material = pred['material']
            if material not in materiales_volumen:
                materiales_volumen[material] = 0
            materiales_volumen[material] += pred['total_predicho']
        
        top_materiales = sorted(materiales_volumen.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (material, volumen) in enumerate(top_materiales, 1):
            material_display = material[:35] + "..." if len(material) > 35 else material
            porcentaje = (volumen / total_predicho) * 100
            print(f"   {i:2d}. {material_display:38} | {volumen:8,.0f} M2 ({porcentaje:4.1f}%)")
        
        # Distribución de precisión
        print(f"\n🎯 DISTRIBUCIÓN DE PRECISIÓN (MAPE):")
        excelente = len([r for r in self.predicciones_detalladas if r['mape'] < 25])
        bueno = len([r for r in self.predicciones_detalladas if 25 <= r['mape'] < 50])
        revisar = len([r for r in self.predicciones_detalladas if r['mape'] >= 50])
        
        print(f"   • Excelente (<25% MAPE): {excelente:,} combinaciones ({excelente/total_combinaciones*100:.1f}%)")
        print(f"   • Bueno (25-50% MAPE): {bueno:,} combinaciones ({bueno/total_combinaciones*100:.1f}%)")
        print(f"   • Revisar (>50% MAPE): {revisar:,} combinaciones ({revisar/total_combinaciones*100:.1f}%)")
        
        # Predicciones mensuales
        print(f"\n📅 PREDICCIONES POR MES:")
        totales_mensuales = [0, 0, 0, 0]
        nombres_meses = ['Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        for pred in self.predicciones_detalladas:
            for i, pred_mes in enumerate(pred['predicciones']):
                if i < 4:
                    totales_mensuales[i] += pred_mes['prediccion']
        
        for i, (mes, total) in enumerate(zip(nombres_meses, totales_mensuales)):
            print(f"   • {mes} 2025: {total:,.0f} M2")
        
        print("\n" + "="*80)
        
        return self
    
    def generar_grafico_tendencias(self, top_combinaciones=10, archivo_grafico='tendencias_forecast.png'):
        """Generar gráfico de tendencias para top combinaciones"""
        
        if not self.predicciones_detalladas:
            print("❌ No hay predicciones disponibles.")
            return
        
        print(f"\n📊 Generando gráfico de tendencias: {archivo_grafico}")
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Forecast por Cliente-Material: Tendencias y Comparativas', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Tendencias de top combinaciones
        ax1.set_title(f'Top {top_combinaciones} Combinaciones: Histórico vs Predicciones', fontsize=14)
        
        colores = plt.cm.Set3(np.linspace(0, 1, top_combinaciones))
        
        for i, pred in enumerate(self.predicciones_detalladas[:top_combinaciones]):
            datos_hist = pred['datos_historicos']
            
            # Datos históricos
            meses_hist = [f"{row['mes']}/{row['año']}" for _, row in datos_hist.iterrows()]
            valores_hist = datos_hist['cantidad_total'].values
            
            # Predicciones
            meses_pred = [f"{p['mes']}/2025" for p in pred['predicciones']]
            valores_pred = [p['prediccion'] for p in pred['predicciones']]
            
            # Combinar para continuidad
            meses_todos = meses_hist + meses_pred
            valores_todos = list(valores_hist) + valores_pred
            
            # Plot
            ax1.plot(range(len(meses_hist)), valores_hist, 'o-', 
                    color=colores[i], linewidth=2, markersize=6, 
                    label=f"{pred['cliente'][:15]}.../{pred['material'][:15]}...")
            
            ax1.plot(range(len(meses_hist)-1, len(meses_todos)), 
                    valores_todos[len(meses_hist)-1:], '--', 
                    color=colores[i], linewidth=2, alpha=0.7)
        
        ax1.axvline(x=len(self.predicciones_detalladas[0]['datos_historicos'])-0.5, 
                   color='red', linestyle=':', alpha=0.7, 
                   label='Inicio Predicciones')
        
        ax1.set_xlabel('Período')
        ax1.set_ylabel('M2')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparativa mensual agregada
        ax2.set_title('Comparativa Mensual: Histórico vs Predicciones (Totales)', fontsize=14)
        
        # Calcular totales mensuales históricos (últimos 4 meses disponibles)
        historicos_mensuales = []
        predichos_mensuales = []
        nombres_meses = []
        
        # Obtener últimos 4 meses históricos para comparar
        meses_disponibles = sorted(self.df_cliente_material['mes'].unique())[-4:]
        
        for mes in meses_disponibles:
            total_mes = self.df_cliente_material[
                self.df_cliente_material['mes'] == mes
            ]['cantidad_total'].sum()
            historicos_mensuales.append(total_mes)
        
        # Totales predichos
        for i in range(4):
            total_pred = sum([
                r['predicciones'][i]['prediccion'] 
                for r in self.predicciones_detalladas 
                if len(r['predicciones']) > i
            ])
            predichos_mensuales.append(total_pred)
            nombres_meses.append(f"Mes {i+1}")
        
        x = np.arange(len(nombres_meses))
        width = 0.35
        
        ax2.bar(x - width/2, historicos_mensuales, width, 
               label='Últimos 4 meses históricos', color='skyblue', alpha=0.8)
        ax2.bar(x + width/2, predichos_mensuales, width, 
               label='Próximos 4 meses predichos', color='lightcoral', alpha=0.8)
        
        # Agregar valores en las barras
        for i, (hist, pred) in enumerate(zip(historicos_mensuales, predichos_mensuales)):
            ax2.text(i - width/2, hist + hist*0.01, f'{hist:,.0f}', 
                    ha='center', va='bottom', fontsize=9)
            ax2.text(i + width/2, pred + pred*0.01, f'{pred:,.0f}', 
                    ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('Período')
        ax2.set_ylabel('M2 Totales')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['May-Ago 2025', 'Jun-Sep 2025', 'Jul-Oct 2025', 'Ago-Nov 2025'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(archivo_grafico, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Gráfico guardado: {archivo_grafico}")
        
        return archivo_grafico
    
    def analisis_comparativo_clientes(self, top_clientes=20):
        """Análisis comparativo detallado por cliente"""
        
        if not self.predicciones_detalladas:
            print("❌ No hay predicciones disponibles.")
            return
        
        print(f"\n🔍 ANÁLISIS COMPARATIVO - TOP {top_clientes} CLIENTES")
        print("="*70)
        
        # Agrupar datos por cliente
        analisis_clientes = {}
        
        for pred in self.predicciones_detalladas:
            cliente = pred['cliente']
            
            if cliente not in analisis_clientes:
                analisis_clientes[cliente] = {
                    'materiales': [],
                    'total_historico': 0,
                    'total_predicho': 0,
                    'mapes': [],
                    'meses_datos': []
                }
            
            analisis_clientes[cliente]['materiales'].append(pred['material'])
            analisis_clientes[cliente]['total_historico'] += pred['total_historico']
            analisis_clientes[cliente]['total_predicho'] += pred['total_predicho']
            analisis_clientes[cliente]['mapes'].append(pred['mape'])
            analisis_clientes[cliente]['meses_datos'].append(pred['meses_datos'])
        
        # Calcular métricas adicionales
        for cliente, data in analisis_clientes.items():
            data['num_materiales'] = len(data['materiales'])
            data['mape_promedio'] = np.mean(data['mapes'])
            data['meses_promedio'] = np.mean(data['meses_datos'])
            data['cambio_vs_historico'] = ((data['total_predicho']/4) / (data['total_historico']/8) - 1) * 100
        
        # Ordenar por volumen predicho
        clientes_ordenados = sorted(
            analisis_clientes.items(), 
            key=lambda x: x[1]['total_predicho'], 
            reverse=True
        )[:top_clientes]
        
        # Mostrar tabla comparativa
        print(f"{'Cliente':<30} | {'Mat.':<4} | {'Hist.Total':<10} | {'Pred.Total':<10} | {'Cambio%':<8} | {'MAPE':<6}")
        print("-" * 70)
        
        for cliente, data in clientes_ordenados:
            cliente_display = cliente[:28] + ".." if len(cliente) > 28 else cliente
            
            print(f"{cliente_display:<30} | "
                  f"{data['num_materiales']:<4} | "
                  f"{data['total_historico']:<10,.0f} | "
                  f"{data['total_predicho']:<10,.0f} | "
                  f"{data['cambio_vs_historico']:>+7.1f}% | "
                  f"{data['mape_promedio']:<6.1f}%")
        
        return analisis_clientes


def main():
    """Función principal para ejecutar el sistema completo"""
    
    print("🚀 SISTEMA DE FORECASTING GRANULAR POR CLIENTE-MATERIAL")
    print("="*60)
    
    # Configuración
    archivo_csv = input("📁 Ingresa la ruta del archivo CSV: ").strip()
    if not archivo_csv:
        archivo_csv = "datos_ventas.csv"  # Archivo por defecto
    
    try:
        # Inicializar sistema
        sistema = ForecastClienteMaterial(archivo_csv)
        
        # Ejecutar pipeline completo
        sistema.cargar_y_preparar_datos()
        sistema.identificar_combinaciones_predictibles(min_meses=3, min_cantidad_total=100)
        sistema.ejecutar_forecast_completo(top_combinaciones=100, meses_forecast=4)
        
        # Generar reportes
        sistema.mostrar_resumen_ejecutivo()
        sistema.generar_reporte_detallado()
        sistema.exportar_excel()
        sistema.generar_grafico_tendencias()
        sistema.analisis_comparativo_clientes()
        
        print(f"\n🎉 PROCESO COMPLETADO EXITOSAMENTE")
        print(f"📊 Archivos generados:")
        print(f"   • forecast_cliente_material.html (reporte web)")
        print(f"   • predicciones_cliente_material.xlsx (datos Excel)")
        print(f"   • tendencias_forecast.png (gráficos)")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()