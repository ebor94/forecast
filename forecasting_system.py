import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

class AdvancedForecastingSystem:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.results_by_method = {}
        self.validation_metrics = {}
        self.df = None
        self.material_hierarchies = {}
        self.client_segments = {}
        self.sales_team_performance = {}
        self.discontinued_materials = set()
        self.substitute_mapping = {}
        
    def load_and_prepare_data(self, file_path=None, data_df=None):
        """
        Carga y prepara los datos con la nueva estructura enriquecida
        """
        if data_df is not None:
            df = data_df.copy()
        else:
            df = pd.read_csv(file_path)
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Crear mapeo de columnas para diferentes nomenclaturas
        column_mapping = {
            'Material': 'SKU' if 'SKU' in df.columns else 'Material',
            'Descripción de posición': 'Descripción de SKU' if 'Descripción de SKU' in df.columns else 'Descripción de posición'
        }
        
        # Convertir fecha a datetime
        df['Fechacompra'] = pd.to_datetime(df['Fechacompra'], format='%m/%d/%Y')
        
        # Extraer características temporales
        df['Año'] = df['Fechacompra'].dt.year
        df['Mes'] = df['Fechacompra'].dt.month
        df['Trimestre'] = df['Fechacompra'].dt.quarter
        df['DiaSemana'] = df['Fechacompra'].dt.dayofweek
        
        # Limpiar datos numéricos
        df['cantidadComprada'] = pd.to_numeric(df['cantidadComprada'], errors='coerce')
        df = df.dropna(subset=['cantidadComprada'])
        
        # Usar SKU como identificador principal si está disponible
        if 'SKU' in df.columns:
            df['Material'] = df['SKU']
        
        # Crear jerarquía de materiales
        self._create_material_hierarchy(df)
        
        # Analizar segmentos de clientes
        self._analyze_client_segments(df)
        
        # Analizar rendimiento por equipo de ventas
        self._analyze_sales_team_performance(df)
        
        # Identificar productos descontinuados
        self._identify_discontinued_products(df)
        
        print(f"✅ Datos enriquecidos cargados: {len(df)} registros")
        print(f"📅 Período: {df['Fechacompra'].min()} a {df['Fechacompra'].max()}")
        print(f"👥 Clientes únicos: {df['IDCliente'].nunique()}")
        print(f"🏗️ Materiales/SKUs únicos: {df['Material'].nunique()}")
        print(f"👨‍💼 Equipos de vendedores: {df['Grupo de vendedores'].nunique() if 'Grupo de vendedores' in df.columns else 0}")
        print(f"📦 Grupos de materiales nivel 1: {df['Grupo de materiales 1'].nunique() if 'Grupo de materiales 1' in df.columns else 0}")
        
        self.df = df
        return df
    
    def _create_material_hierarchy(self, df):
        """
        Crea jerarquía de materiales usando los grupos disponibles
        """
        hierarchy_cols = [col for col in df.columns if 'Grupo de materiales' in col]
        
        if hierarchy_cols:
            # Crear mapeo jerárquico
            for material in df['Material'].unique():
                material_data = df[df['Material'] == material].iloc[0]
                
                hierarchy = {}
                for col in sorted(hierarchy_cols):
                    level = col.replace('Grupo de materiales ', '').strip()
                    hierarchy[f'Nivel_{level}'] = material_data.get(col, 'Sin_Clasificar')
                
                # Agregar descripción si está disponible
                desc_col = 'Descripción de SKU' if 'Descripción de SKU' in df.columns else 'Descripción de posición'
                if desc_col in df.columns:
                    hierarchy['Descripcion'] = material_data.get(desc_col, '')
                
                self.material_hierarchies[material] = hierarchy
        
        print(f"🏗️ Jerarquía de materiales creada para {len(self.material_hierarchies)} productos")
    
    def _analyze_client_segments(self, df):
        """
        Analiza y segmenta clientes basado en patrones de compra
        """
        client_analysis = df.groupby(['IDCliente', 'NombreCliente'] if 'NombreCliente' in df.columns else 'IDCliente').agg({
            'cantidadComprada': ['sum', 'mean', 'count'],
            'Fechacompra': ['min', 'max'],
            'Material': 'nunique',
            'zona de ventas': 'first'
        }).round(2)
        
        client_analysis.columns = ['Total_Comprado', 'Promedio_Compra', 'Num_Transacciones', 
                                   'Primera_Compra', 'Ultima_Compra', 'Materiales_Unicos', 'Zona']
        
        # Segmentar clientes
        client_analysis['Valor_Cliente'] = pd.qcut(client_analysis['Total_Comprado'], 
                                                   q=3, labels=['Bajo', 'Medio', 'Alto'])
        client_analysis['Frecuencia'] = pd.qcut(client_analysis['Num_Transacciones'], 
                                                q=3, labels=['Baja', 'Media', 'Alta'])
        
        # Crear diccionario de segmentos
        for idx, row in client_analysis.iterrows():
            cliente_id = idx[0] if isinstance(idx, tuple) else idx
            self.client_segments[cliente_id] = {
                'valor': row['Valor_Cliente'],
                'frecuencia': row['Frecuencia'],
                'total_comprado': row['Total_Comprado'],
                'materiales_unicos': row['Materiales_Unicos'],
                'zona': row['Zona']
            }
        
        print(f"👥 Segmentación de {len(self.client_segments)} clientes completada")
    
    def _analyze_sales_team_performance(self, df):
        """
        Analiza rendimiento por equipo de ventas
        """
        if 'Grupo de vendedores' not in df.columns:
            return
        
        team_analysis = df.groupby('Grupo de vendedores').agg({
            'cantidadComprada': ['sum', 'mean'],
            'IDCliente': 'nunique',
            'Material': 'nunique',
            'Fechacompra': ['min', 'max']
        }).round(2)
        
        team_analysis.columns = ['Total_Vendido', 'Promedio_Venta', 'Clientes_Unicos', 
                                'Materiales_Vendidos', 'Primera_Venta', 'Ultima_Venta']
        
        # Calcular métricas de rendimiento
        for team, row in team_analysis.iterrows():
            self.sales_team_performance[team] = {
                'total_vendido': row['Total_Vendido'],
                'promedio_venta': row['Promedio_Venta'],
                'clientes_unicos': row['Clientes_Unicos'],
                'productividad': row['Total_Vendido'] / row['Clientes_Unicos'] if row['Clientes_Unicos'] > 0 else 0
            }
        
        print(f"👨‍💼 Análisis de {len(self.sales_team_performance)} equipos de ventas completado")
    
    def _identify_discontinued_products(self, df):
        """
        Identifica productos descontinuados y mapea sustitutos
        """
        # Productos sin ventas en últimos 6 meses
        recent_cutoff = df['Fechacompra'].max() - timedelta(days=180)
        recent_materials = set(df[df['Fechacompra'] >= recent_cutoff]['Material'].unique())
        all_materials = set(df['Material'].unique())
        self.discontinued_materials = all_materials - recent_materials
        
        # Crear mapeo de sustitutos basado en jerarquía
        if 'Grupo de materiales 1' in df.columns:
            for discontinued in self.discontinued_materials:
                if discontinued in self.material_hierarchies:
                    group1 = self.material_hierarchies[discontinued].get('Nivel_1', '')
                    
                    # Buscar materiales activos en el mismo grupo
                    potential_substitutes = []
                    for material, hierarchy in self.material_hierarchies.items():
                        if (material not in self.discontinued_materials and 
                            hierarchy.get('Nivel_1', '') == group1 and 
                            material != discontinued):
                            potential_substitutes.append(material)
                    
                    if potential_substitutes:
                        self.substitute_mapping[discontinued] = potential_substitutes[:5]  # Top 5 sustitutos
        
        print(f"⚠️ Productos descontinuados identificados: {len(self.discontinued_materials)}")
        print(f"🔄 Mapeo de sustitutos creado para: {len(self.substitute_mapping)} productos")
    
    def calculate_validation_metrics(self, actual, predicted):
        """
        Calcula métricas de validación: MAD, MAPE, ACC
        """
        # Eliminar valores nulos o infinitos
        mask = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {'MAD': np.nan, 'MAPE': np.nan, 'ACC': np.nan}
        
        # MAD (Mean Absolute Deviation)
        mad = np.mean(np.abs(actual_clean - predicted_clean))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        
        # ACC (Accuracy) = 1 - MAPE/100
        acc = 1 - (mape / 100)
        
        return {
            'MAD': round(mad, 4),
            'MAPE': round(mape, 2),
            'ACC': round(acc, 4)
        }
    
    def moving_average_forecast(self, data, window=3, periods_ahead=3):
        """Método de Promedio Móvil"""
        if len(data) < window:
            avg = data.mean() if len(data) > 0 else 0
            return [avg] * periods_ahead
        
        moving_avg = data.rolling(window=window, min_periods=1).mean()
        last_ma = moving_avg.iloc[-1]
        return [last_ma] * periods_ahead
    
    def exponential_smoothing_forecast(self, data, alpha=0.3, periods_ahead=3):
        """Método de Suavización Exponencial Simple"""
        if len(data) == 0:
            return [0] * periods_ahead
        if len(data) == 1:
            return [data.iloc[0]] * periods_ahead
        
        smoothed = [data.iloc[0]]
        for i in range(1, len(data)):
            smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[i-1])
        
        last_smoothed = smoothed[-1]
        return [last_smoothed] * periods_ahead
    
    def linear_regression_forecast(self, data, periods_ahead=3):
        """Método de Regresión Lineal Simple"""
        if len(data) < 2:
            avg = data.mean() if len(data) > 0 else 0
            return [avg] * periods_ahead
        
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_X = np.arange(len(data), len(data) + periods_ahead).reshape(-1, 1)
        forecasts = model.predict(future_X)
        
        return np.maximum(forecasts, 0).tolist()
    
    def _multiple_regression_forecast(self, data: pd.DataFrame, periods_ahead: int = 3) -> list:
        """
        Método de Regresión Múltiple con variables enriquecidas, adaptado para manejar
        un número dinámico de períodos futuros.
        """
        if len(data) < 3:
            avg = data['cantidadcomprada'].mean() if 'cantidadcomprada' in data.columns else 0
            return [max(0, avg)] * periods_ahead

        data = data.copy()
        data['time_idx'] = range(len(data))
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        client = data['idcliente'].iloc[0]
        if client in self.client_segments:
            segment = self.client_segments[client]
            data['valor_cliente'] = 1 if segment.get('valor_cliente') == 'Alto' else (0.5 if segment.get('valor_cliente') == 'Medio' else 0)
            data['frecuencia_cliente'] = 1 if segment.get('frecuencia_cliente') == 'Alta' else (0.5 if segment.get('frecuencia_cliente') == 'Media' else 0)
        else:
            data['valor_cliente'] = 0.5
            data['frecuencia_cliente'] = 0.5
        
        # Agregue el nombre de la característica para garantizar un orden consistente
        features = ['time_idx', 'month', 'month_sin', 'month_cos', 'valor_cliente', 'frecuencia_cliente']
        
        X = data[features].values
        y = data['cantidadcomprada'].values
        
        model = LinearRegression().fit(X, y)
        
        future_features = []
        for i in range(periods_ahead):
            # Calcule el mes siguiente de manera dinámica
            next_month = (data['month'].iloc[-1] + i) % 12
            if next_month == 0: next_month = 12
            
            future_features.append([
                len(data) + i, 
                next_month, 
                np.sin(2 * np.pi * next_month / 12), 
                np.cos(2 * np.pi * next_month / 12), 
                data['valor_cliente'].iloc[0], 
                data['frecuencia_cliente'].iloc[0]
            ])
        
        # Convierta la lista de listas en un arreglo de NumPy antes de predecir
        future_features = np.array(future_features)
        forecasts = model.predict(future_features)
        
        return np.maximum(forecasts, 0).tolist()
    
    def get_enhanced_material_history(self, cliente, material):
        """
        Obtiene historial enriquecido considerando jerarquía y sustitutos
        """
        df = self.df.copy()
        
        # Historial directo del material
        direct_history = df[(df['IDCliente'] == cliente) & (df['Material'] == material)]
        
        # Si el material está descontinuado o tiene pocos datos
        if len(direct_history) < 3 or material in self.discontinued_materials:
            
            # Opción 1: Usar sustitutos mapeados
            if material in self.substitute_mapping:
                substitute_materials = self.substitute_mapping[material]
                substitute_history = df[
                    (df['IDCliente'] == cliente) & 
                    (df['Material'].isin(substitute_materials))
                ]
                
                if len(substitute_history) > len(direct_history):
                    print(f"  🔄 Usando datos de sustitutos para {material}")
                    return substitute_history.sort_values('Fechacompra')
            
            # Opción 2: Usar misma jerarquía de materiales
            if material in self.material_hierarchies:
                material_group = self.material_hierarchies[material].get('Nivel_1', '')
                
                # Buscar materiales en el mismo grupo
                same_group_materials = []
                for mat, hierarchy in self.material_hierarchies.items():
                    if hierarchy.get('Nivel_1', '') == material_group:
                        same_group_materials.append(mat)
                
                if same_group_materials:
                    group_history = df[
                        (df['IDCliente'] == cliente) & 
                        (df['Material'].isin(same_group_materials))
                    ]
                    
                    if len(group_history) > len(direct_history):
                        print(f"  📦 Usando datos del grupo '{material_group}' para {material}")
                        return group_history.sort_values('Fechacompra')
        
        return direct_history.sort_values('Fechacompra')
    
    def calculate_enhanced_seasonal_factor(self, group_data, target_month):
        """
        Calcula factor estacional enriquecido considerando múltiples variables
        """
        if len(group_data) < 4:
            return 1.15 if target_month in [11, 12] else 1.0
        
        # Factor base estacional
        monthly_avg = group_data.groupby('Mes')['cantidadComprada'].mean()
        overall_avg = group_data['cantidadComprada'].mean()
        
        base_factor = 1.0
        if target_month in monthly_avg.index and overall_avg > 0:
            base_factor = monthly_avg[target_month] / overall_avg
        
        # Ajuste por equipo de vendedores si está disponible
        team_factor = 1.0
        if 'Grupo de vendedores' in group_data.columns:
            team = group_data['Grupo de vendedores'].iloc[0]
            if team in self.sales_team_performance:
                # Equipos con mayor productividad tienden a vender más en Q4
                productivity = self.sales_team_performance[team]['productividad']
                avg_productivity = np.mean([t['productividad'] for t in self.sales_team_performance.values()])
                if avg_productivity > 0:
                    team_factor = 1 + (productivity - avg_productivity) / avg_productivity * 0.1
        
        # Factor combinado
        combined_factor = base_factor * team_factor
        return max(0.5, min(2.5, combined_factor))
    
    def validate_model_performance(self, client_material_data, method='moving_average', test_ratio=0.3):
        """Valida el rendimiento del modelo usando datos históricos"""
        if len(client_material_data) < 4:
            return {'MAD': np.nan, 'MAPE': np.nan, 'ACC': np.nan}
        
        split_point = int(len(client_material_data) * (1 - test_ratio))
        train_data = client_material_data.iloc[:split_point]
        test_data = client_material_data.iloc[split_point:]
        
        if len(test_data) == 0:
            return {'MAD': np.nan, 'MAPE': np.nan, 'ACC': np.nan}
        
        # Hacer predicciones según el método
        if method == 'moving_average':
            predictions = self.moving_average_forecast(
                train_data['cantidadComprada'], 
                window=min(3, len(train_data)), 
                periods_ahead=len(test_data)
            )
        elif method == 'exponential_smoothing':
            predictions = self.exponential_smoothing_forecast(
                train_data['cantidadComprada'], 
                periods_ahead=len(test_data)
            )
        elif method == 'linear_regression':
            predictions = self.linear_regression_forecast(
                train_data['cantidadComprada'], 
                periods_ahead=len(test_data)
            )
        elif method == 'multiple_regression':
            predictions = self.multiple_regression_forecast(
                train_data, 
                periods_ahead=len(test_data)
            )
        else:
            return {'MAD': np.nan, 'MAPE': np.nan, 'ACC': np.nan}
        
        predictions = predictions[:len(test_data)]
        actual_values = test_data['cantidadComprada'].values
        
        return self.calculate_validation_metrics(actual_values, np.array(predictions))
    
    def forecast_q4_all_methods(self, target_year=2025):
        """
        Genera forecasts para Q4 usando todos los métodos con datos enriquecidos
        """
        df = self.df.copy()
        methods = ['moving_average', 'exponential_smoothing', 'linear_regression', 'multiple_regression']
        
        # Inicializar resultados
        for method in methods:
            self.results_by_method[method] = []
            self.validation_metrics[method] = []
        
        # Obtener combinaciones únicas de cliente-material
        combinations = df.groupby(['IDCliente', 'Material'])
        
        print(f"🔄 Procesando {len(combinations)} combinaciones con datos enriquecidos...")
        
        for (cliente, material), group_data in combinations:
            # Obtener historial enriquecido
            enhanced_history = self.get_enhanced_material_history(cliente, material)
            
            if len(enhanced_history) < 2:
                continue
                
            # Información base enriquecida
            zona = group_data['zona de ventas'].iloc[-1]
            ultima_compra = enhanced_history['Fechacompra'].max()
            
            # Información de jerarquía
            hierarchy_info = self.material_hierarchies.get(material, {})
            grupo_material_1 = hierarchy_info.get('Nivel_1', 'Sin_Clasificar')
            grupo_material_3 = hierarchy_info.get('Nivel_3', 'Sin_Clasificar')
            grupo_material_4 = hierarchy_info.get('Nivel_4', 'Sin_Clasificar')
            
            # Información del cliente
            client_info = self.client_segments.get(cliente, {})
            
            # Información del vendedor
            vendedor = group_data['Grupo de vendedores'].iloc[0] if 'Grupo de vendedores' in group_data.columns else 'Sin_Asignar'
            
            # Procesar cada método
            for method in methods:
                try:
                    # Validar rendimiento
                    validation_metrics = self.validate_model_performance(enhanced_history, method)
                    
                    # Generar predicciones
                    if method == 'moving_average':
                        predictions = self.moving_average_forecast(
                            enhanced_history['cantidadComprada'], 
                            window=min(3, len(enhanced_history))
                        )
                    elif method == 'exponential_smoothing':
                        predictions = self.exponential_smoothing_forecast(
                            enhanced_history['cantidadComprada']
                        )
                    elif method == 'linear_regression':
                        predictions = self.linear_regression_forecast(
                            enhanced_history['cantidadComprada']
                        )
                    elif method == 'multiple_regression':
                         predictions = self._multiple_regression_forecast(enhanced_history, periods_ahead=self.FORECAST_PERIODS)

                    
                    # Guardar resultados para cada mes del Q4
                    for i, mes in enumerate([10, 11, 12]):
                        if i < len(predictions):
                            # Aplicar factor estacional enriquecido
                            seasonal_factor = self.calculate_enhanced_seasonal_factor(enhanced_history, mes)
                            adjusted_prediction = predictions[i] * seasonal_factor
                            
                            result = {
                                'IDCliente': cliente,
                                'NombreCliente': group_data['NombreCliente'].iloc[0] if 'NombreCliente' in group_data.columns else '',
                                'Material': material,
                                'SKU': material,  # Mantener compatibilidad
                                'Descripcion': hierarchy_info.get('Descripcion', ''),
                                'Grupo_Material_1': grupo_material_1,
                                'Grupo_Material_3': grupo_material_3, 
                                'Grupo_Material_4': grupo_material_4,
                                'Grupo_Vendedores': vendedor,
                                'zona_ventas': zona,
                                'Mes_prediccion': mes,
                                'Trimestre': 4,
                                'Año': target_year,
                                'cantidad_sugerida': round(adjusted_prediction, 3),
                                'metodo': method,
                                'es_descontinuado': material in self.discontinued_materials,
                                'tiene_sustitutos': material in self.substitute_mapping,
                                'sustitutos': self.substitute_mapping.get(material, [])[:3],
                                'datos_usados': 'enriquecidos' if len(enhanced_history) > len(group_data) else 'directos',
                                'segmento_cliente': client_info.get('valor', 'Sin_Segmentar'),
                                'frecuencia_cliente': client_info.get('frecuencia', 'Sin_Datos'),
                                'ultima_compra': ultima_compra.strftime('%Y-%m-%d'),
                                'registros_historicos': len(enhanced_history),
                                'registros_directos': len(group_data),
                                'promedio_historico': round(enhanced_history['cantidadComprada'].mean(), 3),
                                'factor_estacional': round(seasonal_factor, 3),
                                'MAD': validation_metrics['MAD'],
                                'MAPE': validation_metrics['MAPE'],
                                'ACC': validation_metrics['ACC']
                            }
                            self.results_by_method[method].append(result)
                    
                    # Guardar métricas de validación
                    validation_record = {
                        'IDCliente': cliente,
                        'Material': material,
                        'Grupo_Material_1': grupo_material_1,
                        'metodo': method,
                        'MAD': validation_metrics['MAD'],
                        'MAPE': validation_metrics['MAPE'],
                        'ACC': validation_metrics['ACC']
                    }
                    self.validation_metrics[method].append(validation_record)
                    
                except Exception as e:
                    print(f"⚠️ Error procesando {cliente}-{material} con {method}: {str(e)}")
                    continue
        
        print("✅ Forecasting enriquecido completado para todos los métodos")
        return self.results_by_method
    
    def generate_enhanced_reports(self):
        """
        Genera reportes enriquecidos con toda la información disponible
        """
        if not self.results_by_method:
            print("❌ No hay resultados para generar reportes")
            return None
        
        # Determinar mejor método
        best_method = self._get_best_method_name()
        if not best_method:
            return None
        
        df_results = pd.DataFrame(self.results_by_method[best_method])
        
        print("\n" + "="*80)
        print("📊 REPORTE ENRIQUECIDO DE FORECASTING Q4")
        print("="*80)
        
        # Análisis por Grupo de Materiales Nivel 1
        print(f"\n🏗️ ANÁLISIS POR GRUPO DE MATERIALES (NIVEL 1):")
        group_analysis = df_results.groupby('Grupo_Material_1').agg({
            'cantidad_sugerida': ['sum', 'mean', 'count'],
            'IDCliente': 'nunique',
            'Material': 'nunique',
            'es_descontinuado': 'sum'
        }).round(2)
        
        group_analysis.columns = ['Total_Sugerido', 'Promedio', 'Predicciones', 'Clientes', 'Materiales', 'Descontinuados']
        group_analysis = group_analysis.sort_values('Total_Sugerido', ascending=False)
        print(group_analysis.to_string())
        
        # Análisis por Segmento de Cliente
        print(f"\n👥 ANÁLISIS POR SEGMENTO DE CLIENTE:")
        segment_analysis = df_results.groupby('segmento_cliente')['cantidad_sugerida'].agg(['sum', 'mean', 'count']).round(2)
        segment_analysis.columns = ['Total_Sugerido', 'Promedio_Sugerido', 'Num_Predicciones']
        print(segment_analysis.to_string())
        
        # Análisis por Equipo de Vendedores
        if 'Grupo_Vendedores' in df_results.columns:
            print(f"\n👨‍💼 ANÁLISIS POR EQUIPO DE VENDEDORES:")
            sales_analysis = df_results.groupby('Grupo_Vendedores')['cantidad_sugerida'].agg(['sum', 'mean', 'count']).round(2)
            sales_analysis.columns = ['Total_Sugerido', 'Promedio_Sugerido', 'Num_Predicciones']
            sales_analysis = sales_analysis.sort_values('Total_Sugerido', ascending=False)
            print(sales_analysis.head(10).to_string())
        
        # Productos descontinuados con recomendaciones
        print(f"\n⚠️ PRODUCTOS DESCONTINUADOS CON PREDICCIONES:")
        discontinued = df_results[df_results['es_descontinuado'] == True]
        if len(discontinued) > 0:
            for _, row in discontinued.head(10).iterrows():
                print(f"   • Material {row['Material']} - Cliente: {row['NombreCliente']}")
                print(f"     Grupo: {row['Grupo_Material_1']} | Predicción Q4: {row['cantidad_sugerida']}")
                if row['sustitutos']:
                    print(f"     Sustitutos sugeridos: {row['sustitutos']}")
                print()
        else:
            print("   ✅ No hay productos descontinuados con predicciones activas")
        
        # Top oportunidades por cliente
        print(f"\n🎯 TOP OPORTUNIDADES POR CLIENTE:")
        client_opportunities = df_results.groupby(['IDCliente', 'NombreCliente'])['cantidad_sugerida'].sum().sort_values(ascending=False)
        for (cliente_id, nombre), total in client_opportunities.head(10).items():
            segment_info = self.client_segments.get(cliente_id, {})
            print(f"   • {nombre} (ID: {cliente_id})")
            print(f"     Total Q4 proyectado: {total:,.2f}")
            print(f"     Segmento: {segment_info.get('valor', 'N/A')} | Frecuencia: {segment_info.get('frecuencia', 'N/A')}")
            print()
        
        return {
            'group_analysis': group_analysis,
            'segment_analysis': segment_analysis,
            'client_opportunities': client_opportunities,
            'discontinued_products': discontinued
        }
    
    def _get_best_method_name(self):
        """Determina el mejor método basado en ACC promedio"""
        best_method = None
        best_acc = 0
        
        for method, metrics_list in self.validation_metrics.items():
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                df_metrics = df_metrics.dropna(subset=['ACC'])
                if len(df_metrics) > 0:
                    avg_acc = df_metrics['ACC'].mean()
                    if avg_acc > best_acc:
                        best_acc = avg_acc
                        best_method = method
        
        return best_method
    
    def compare_methods_performance(self):
        """Compara el rendimiento de todos los métodos"""
        if not self.validation_metrics:
            print("❌ No hay métricas de validación disponibles")
            return None
        
        comparison_results = []
        
        for method, metrics_list in self.validation_metrics.items():
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                df_metrics = df_metrics.dropna(subset=['MAD', 'MAPE', 'ACC'])
                
                if len(df_metrics) > 0:
                    avg_metrics = {
                        'Método': method.replace('_', ' ').title(),
                        'Promedio_MAD': round(df_metrics['MAD'].mean(), 4),
                        'Promedio_MAPE': round(df_metrics['MAPE'].mean(), 2),
                        'Promedio_ACC': round(df_metrics['ACC'].mean(), 4),
                        'Casos_validados': len(df_metrics),
                        'Mediana_MAD': round(df_metrics['MAD'].median(), 4),
                        'Mediana_MAPE': round(df_metrics['MAPE'].median(), 2),
                        'Mediana_ACC': round(df_metrics['ACC'].median(), 4)
                    }
                    comparison_results.append(avg_metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('Promedio_ACC', ascending=False)
            
            print("\n" + "="*80)
            print("📊 COMPARACIÓN DE RENDIMIENTO DE MÉTODOS")
            print("="*80)
            print(comparison_df.to_string(index=False))
            
            best_method = comparison_df.iloc[0]
            print(f"\n🏆 MEJOR MÉTODO: {best_method['Método']}")
            print(f"   • Precisión (ACC): {best_method['Promedio_ACC']:.4f}")
            print(f"   • MAPE: {best_method['Promedio_MAPE']:.2f}%")
            print(f"   • MAD: {best_method['Promedio_MAD']:.4f}")
        
        return comparison_df
    
    def export_enhanced_results(self, base_filename='forecasting_enhanced'):
        """Exporta todos los resultados enriquecidos"""
        # Exportar resultados por método
        for method, results in self.results_by_method.items():
            if results:
                df_results = pd.DataFrame(results)
                filename = f"{base_filename}_{method}.csv"
                df_results.to_csv(filename, index=False, encoding='utf-8-sig')
                print(f"✅ Resultados {method} exportados a: {filename}")
        
        # Exportar comparación de métodos
        comparison_df = self.compare_methods_performance()
        if comparison_df is not None:
            comparison_df.to_csv(f"{base_filename}_comparison.csv", index=False, encoding='utf-8-sig')
            print(f"✅ Comparación exportada a: {base_filename}_comparison.csv")
        
        # Exportar análisis de segmentos de clientes
        if self.client_segments:
            client_df = pd.DataFrame.from_dict(self.client_segments, orient='index')
            client_df.to_csv(f"{base_filename}_client_segments.csv", encoding='utf-8-sig')
            print(f"✅ Segmentos de clientes exportados a: {base_filename}_client_segments.csv")
        
        # Exportar análisis de equipos de ventas
        if self.sales_team_performance:
            sales_df = pd.DataFrame.from_dict(self.sales_team_performance, orient='index')
            sales_df.to_csv(f"{base_filename}_sales_teams.csv", encoding='utf-8-sig')
            print(f"✅ Análisis de equipos exportado a: {base_filename}_sales_teams.csv")
        
        # Exportar mapeo de sustitutos
        if self.substitute_mapping:
            substitute_data = []
            for discontinued, substitutes in self.substitute_mapping.items():
                for substitute in substitutes:
                    substitute_data.append({
                        'Material_Descontinuado': discontinued,
                        'Material_Sustituto': substitute,
                        'Grupo_Material': self.material_hierarchies.get(discontinued, {}).get('Nivel_1', 'Sin_Grupo')
                    })
            
            substitute_df = pd.DataFrame(substitute_data)
            substitute_df.to_csv(f"{base_filename}_substitutes.csv", index=False, encoding='utf-8-sig')
            print(f"✅ Mapeo de sustitutos exportado a: {base_filename}_substitutes.csv")
    
    def get_best_method_results(self):
        """Retorna los resultados del mejor método"""
        best_method_name = self._get_best_method_name()
        
        if best_method_name and best_method_name in self.results_by_method:
            best_results = pd.DataFrame(self.results_by_method[best_method_name])
            print(f"✅ Retornando resultados del mejor método: {best_method_name}")
            return best_results
        else:
            print(f"❌ No se encontraron resultados válidos")
            return None
    
    def run_comprehensive_analysis(self, file_path=None, data_df=None):
        """
        Ejecuta análisis completo con datos enriquecidos
        """
        print("🚀 INICIANDO ANÁLISIS COMPRENSIVO ENRIQUECIDO")
        print("="*60)
        
        # 1. Cargar y preparar datos enriquecidos
        self.load_and_prepare_data(file_path, data_df)
        
        # 2. Ejecutar forecasting con todos los métodos
        self.forecast_q4_all_methods()
        
        # 3. Comparar métodos
        self.compare_methods_performance()
        
        # 4. Generar reportes enriquecidos
        enhanced_reports = self.generate_enhanced_reports()
        
        # 5. Exportar todos los resultados
        self.export_enhanced_results()
        
        # 6. Retornar resultados del mejor método
        best_results = self.get_best_method_results()
        
        print("\n✅ ANÁLISIS COMPRENSIVO ENRIQUECIDO COMPLETADO!")
        
        return {
            'best_predictions': best_results,
            'enhanced_reports': enhanced_reports,
            'all_methods': self.results_by_method,
            'client_segments': self.client_segments,
            'sales_team_performance': self.sales_team_performance,
            'substitute_mapping': self.substitute_mapping
        }

# Funciones simplificadas para uso directo
def comprehensive_forecast_enhanced(file_path=None, data_df=None):
    """
    Función simplificada para ejecutar forecasting enriquecido completo
    """
    forecaster = AdvancedForecastingSystem()
    return forecaster.run_comprehensive_analysis(file_path, data_df)

def analyze_data_structure(data_df):
    """
    Analiza la estructura de datos enriquecida y muestra información clave
    """
    print("🔍 ANÁLISIS DE ESTRUCTURA DE DATOS ENRIQUECIDA")
    print("="*60)
    
    print(f"📊 Dimensiones: {data_df.shape[0]} filas x {data_df.shape[1]} columnas")
    print(f"📅 Rango de fechas: {data_df['Fechacompra'].min()} a {data_df['Fechacompra'].max()}")
    
    # Análisis de columnas clave
    key_columns = {
        'IDCliente': 'Clientes únicos',
        'NombreCliente': 'Nombres de clientes',
        'Material': 'Materiales/SKUs únicos',
        'SKU': 'SKUs únicos',
        'Grupo de materiales 1': 'Grupos de materiales (Nivel 1)',
        'Grupo de materiales 3': 'Grupos de materiales (Nivel 3)',
        'Grupo de materiales 4': 'Grupos de materiales (Nivel 4)',
        'Grupo de vendedores': 'Equipos de vendedores',
        'zona de ventas': 'Zonas de ventas'
    }
    
    print(f"\n📈 RESUMEN DE DATOS ÚNICOS:")
    for col, description in key_columns.items():
        if col in data_df.columns:
            unique_count = data_df[col].nunique()
            print(f"   • {description}: {unique_count}")
    
    # Top grupos de materiales
    if 'Grupo de materiales 1' in data_df.columns:
        print(f"\n🏗️ TOP GRUPOS DE MATERIALES (NIVEL 1):")
        top_groups = data_df.groupby('Grupo de materiales 1')['cantidadComprada'].sum().sort_values(ascending=False).head(5)
        for group, total in top_groups.items():
            print(f"   • {group}: {total:,.2f}")
    
    # Top clientes
    if 'NombreCliente' in data_df.columns:
        print(f"\n👥 TOP CLIENTES POR VOLUMEN:")
        top_clients = data_df.groupby('NombreCliente')['cantidadComprada'].sum().sort_values(ascending=False).head(5)
        for client, total in top_clients.items():
            print(f"   • {client}: {total:,.2f}")
    
    # Análisis temporal
    print(f"\n📅 DISTRIBUCIÓN TEMPORAL:")
    monthly_dist = data_df.groupby(data_df['Fechacompra'].dt.to_period('M'))['cantidadComprada'].sum().tail(6)
    for period, total in monthly_dist.items():
        print(f"   • {period}: {total:,.2f}")

