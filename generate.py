import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set Plotly template
pio.templates.default = "plotly_white"

class EnhancedDashboardGenerator:
    """Generate interactive dashboard for enhanced agricultural forecasts"""
    
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.colors = {
            'national': '#2E8B57',  # Sea Green
            'regional': '#4682B4',  # Steel Blue
            'positive': '#00AA00',
            'negative': '#FF4444',
            'neutral': '#FFAA00'
        }
    
    def create_comprehensive_dashboard(self, historical_data, forecasts, output_file="enhanced_agriculture_dashboard.html"):
        """Create an enhanced interactive Plotly dashboard"""
        
        # Create dashboard with proper subplot configuration
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=(
                'National Agriculture GDP Trend & Forecast',
                'Feature Importance Analysis',
                'Model Performance Comparison',
                'Regional Forecast Comparison',
                'Growth Rate Analysis',
                'Forecast Confidence Intervals',
                'Historical Data Distribution',
                'Year-over-Year Changes',
                'Data Completeness Analysis',
                'Economic Cycle Impact',
                'Uncertainty Trend Analysis',
                'Farmer Recommendation Summary'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}]  # Only first subplot in last row needs secondary_y
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.08
        )
        
        # Add all visualizations
        self._add_national_trend_forecast(fig, historical_data, forecasts, row=1, col=1)
        self._add_feature_importance_analysis(fig, row=1, col=2)
        self._add_model_performance_comparison(fig, row=1, col=3)
        self._add_regional_forecast_comparison(fig, forecasts, row=2, col=1)
        self._add_growth_rate_analysis(fig, forecasts, row=2, col=2)
        self._add_confidence_intervals_analysis(fig, forecasts, row=2, col=3)
        self._add_historical_distribution(fig, historical_data, row=3, col=1)
        self._add_yearly_change_analysis(fig, historical_data, row=3, col=2)
        self._add_data_completeness_analysis(fig, historical_data, row=3, col=3)
        self._add_economic_cycle_analysis(fig, historical_data, row=4, col=1)
        self._add_uncertainty_trend_analysis(fig, forecasts, row=4, col=2)
        self._add_recommendation_summary(fig, forecasts, row=4, col=3)
        
        # Update layout
        fig.update_layout(
            title_text="🌾 Myanmar Agricultural Forecasting - Comprehensive Dashboard",
            title_x=0.5,
            title_font_size=24,
            height=1600,
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # Save as HTML
        fig.write_html(output_file)
        print(f"✅ Enhanced dashboard saved as {output_file}")
        
        return fig
    
    def _add_national_trend_forecast(self, fig, historical_data, forecasts, row, col):
        """Add national trend and forecast visualization"""
        if historical_data and 'national' in historical_data and forecasts and 'national' in forecasts:
            national_data = historical_data['national']
            national_forecast = forecasts['national']
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=national_data['Year'],
                    y=national_data['Agriculture_Value_Added'],
                    mode='lines+markers',
                    name='Historical Data',
                    line=dict(color=self.colors['national'], width=4),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Value:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Forecast data
            forecast_years = [item['year'] for item in national_forecast]
            forecast_means = [item['mean'] for item in national_forecast]
            forecast_lower = [item['lower'] for item in national_forecast]
            forecast_upper = [item['upper'] for item in national_forecast]
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=forecast_years + forecast_years[::-1],
                    y=forecast_upper + forecast_lower[::-1],
                    fill='toself',
                    fillcolor='rgba(46, 139, 87, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='80% Confidence Interval',
                    showlegend=True,
                    hovertemplate='<b>Year:</b> %{x}<br><b>Range:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Mean forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_years,
                    y=forecast_means,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3, dash='dash'),
                    marker=dict(size=8, symbol='star'),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Forecast:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add trend line for historical data
            if len(national_data) > 1:
                z = np.polyfit(national_data['Year'], national_data['Agriculture_Value_Added'], 1)
                trend_line = np.poly1d(z)(national_data['Year'])
                
                fig.add_trace(
                    go.Scatter(
                        x=national_data['Year'],
                        y=trend_line,
                        mode='lines',
                        name='Historical Trend',
                        line=dict(color='orange', width=2, dash='dot'),
                        hovertemplate='<b>Trend</b><extra></extra>'
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Year", row=row, col=col)
        fig.update_yaxes(title_text="Agriculture Value Added (% of GDP)", row=row, col=col)
    
    def _add_feature_importance_analysis(self, fig, row, col):
        """Add feature importance analysis"""
        if hasattr(self.forecaster, 'feature_importance') and 'national' in self.forecaster.feature_importance:
            importance_df = self.forecaster.feature_importance['national'].head(15)
            
            # Create custom color scale based on importance
            colorscale = px.colors.sequential.Viridis
            
            fig.add_trace(
                go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    marker_color=importance_df['importance'],
                    marker_colorscale=colorscale,
                    name='Feature Importance',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Importance Score", row=row, col=col)
            fig.update_yaxes(title_text="Features", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Feature Importance Data")
    
    def _add_model_performance_comparison(self, fig, row, col):
        """Add model performance comparison"""
        if hasattr(self.forecaster, 'performance_metrics'):
            metrics = self.forecaster.performance_metrics
            
            models = list(metrics.keys())
            r2_scores = [metrics[model]['test_r2'] for model in models]
            rmse_scores = [metrics[model]['test_rmse'] for model in models]
            
            # Create separate traces for each metric
            fig.add_trace(
                go.Bar(
                    name='R² Score',
                    x=models,
                    y=r2_scores,
                    marker_color='lightblue',
                    hovertemplate='<b>%{x}</b><br>R² Score: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add RMSE as a line plot on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    name='RMSE',
                    x=models,
                    y=rmse_scores,
                    mode='lines+markers',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    yaxis='y2',
                    hovertemplate='<b>%{x}</b><br>RMSE: %{y:.4f}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add secondary y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='RMSE',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
            
            fig.update_xaxes(title_text="Models", row=row, col=col)
            fig.update_yaxes(title_text="R² Score", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Performance Metrics")
    
    def _add_regional_forecast_comparison(self, fig, forecasts, row, col):
        """Add regional forecast comparison"""
        regional_forecasts = {k: v for k, v in forecasts.items() if k != 'national'}
        
        if regional_forecasts:
            colors = px.colors.qualitative.Set3
            
            for i, (region, forecast_data) in enumerate(regional_forecasts.items()):
                years = [item['year'] for item in forecast_data]
                means = [item['mean'] for item in forecast_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=means,
                        mode='lines+markers',
                        name=region,
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{region}</b><br>Year: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Agriculture Value", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Regional Forecasts")
    
    def _add_growth_rate_analysis(self, fig, forecasts, row, col):
        """Add growth rate analysis"""
        if 'national' in forecasts:
            forecast_data = forecasts['national']
            years = [item['year'] for item in forecast_data]
            means = [item['mean'] for item in forecast_data]
            
            growth_rates = []
            for i in range(1, len(means)):
                growth = ((means[i] - means[i-1]) / means[i-1]) * 100 if means[i-1] != 0 else 0
                growth_rates.append(growth)
            
            # Create color array based on growth values
            colors = [self.colors['positive'] if rate > 0 else self.colors['negative'] for rate in growth_rates]
            
            fig.add_trace(
                go.Bar(
                    x=years[1:],
                    y=growth_rates,
                    marker_color=colors,
                    name='Growth Rate',
                    hovertemplate='<b>Year:</b> %{x}<br><b>Growth:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=row, col=col)
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Growth Rate (%)", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Growth Rate Data")
    
    def _add_confidence_intervals_analysis(self, fig, forecasts, row, col):
        """Add confidence intervals analysis"""
        if 'national' in forecasts:
            forecast_data = forecasts['national']
            years = [item['year'] for item in forecast_data]
            confidences = [item['confidence'] for item in forecast_data]
            means = [item['mean'] for item in forecast_data]
            
            # Calculate relative uncertainty (confidence / mean)
            relative_uncertainty = [(conf / mean * 100) if mean != 0 else 0 for conf, mean in zip(confidences, means)]
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=relative_uncertainty,
                    mode='lines+markers',
                    name='Relative Uncertainty',
                    line=dict(color='purple', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Uncertainty:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Relative Uncertainty (%)", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Uncertainty Data")
    
    def _add_historical_distribution(self, fig, historical_data, row, col):
        """Add historical data distribution"""
        if historical_data and 'national' in historical_data:
            national_data = historical_data['national']
            values = national_data['Agriculture_Value_Added']
            
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=20,
                    name='Value Distribution',
                    marker_color=self.colors['national'],
                    opacity=0.7,
                    hovertemplate='<b>Value:</b> %{x:.2f}%<br><b>Count:</b> %{y}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add mean and median lines
            mean_val = values.mean()
            median_val = values.median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                         annotation_text=f"Mean: {mean_val:.2f}%", row=row, col=col)
            fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                         annotation_text=f"Median: {median_val:.2f}%", row=row, col=col)
            
            fig.update_xaxes(title_text="Agriculture Value Added (% of GDP)", row=row, col=col)
            fig.update_yaxes(title_text="Frequency", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Historical Distribution")
    
    def _add_yearly_change_analysis(self, fig, historical_data, row, col):
        """Add yearly change analysis"""
        if historical_data and 'national' in historical_data:
            national_data = historical_data['national'].sort_values('Year')
            national_data['Yearly_Change'] = national_data['Agriculture_Value_Added'].pct_change() * 100
            
            # Remove first row with NaN
            change_data = national_data.iloc[1:]
            
            colors = [self.colors['positive'] if x > 0 else self.colors['negative'] for x in change_data['Yearly_Change']]
            
            fig.add_trace(
                go.Bar(
                    x=change_data['Year'],
                    y=change_data['Yearly_Change'],
                    marker_color=colors,
                    name='Yearly Change',
                    hovertemplate='<b>Year:</b> %{x}<br><b>Change:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=row, col=col)
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Yearly Change (%)", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Yearly Change Data")
    
    def _add_data_completeness_analysis(self, fig, historical_data, row, col):
        """Add data completeness analysis"""
        datasets = ['National', 'Union', 'State_Region', 'Township']
        completeness = []
        
        for key in ['national', 'union', 'state_region', 'township']:
            if historical_data.get(key) is not None:
                df = historical_data[key]
                # Simple completeness measure
                comp = min(100, df.count().sum() / (len(df.columns) * len(df)) * 100)
                completeness.append(comp)
            else:
                completeness.append(0)
        
        colors = [self.colors['positive'] if x > 80 else self.colors['neutral'] if x > 50 else self.colors['negative'] for x in completeness]
        
        fig.add_trace(
            go.Bar(
                x=datasets,
                y=completeness,
                marker_color=colors,
                name='Data Completeness',
                hovertemplate='<b>%{x}</b><br>Completeness: %{y:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Dataset", row=row, col=col)
        fig.update_yaxes(title_text="Completeness (%)", row=row, col=col, range=[0, 100])
    
    def _add_economic_cycle_analysis(self, fig, historical_data, row, col):
        """Add economic cycle analysis"""
        if historical_data and 'national' in historical_data:
            national_data = historical_data['national']
            
            # Create economic cycle feature (simplified)
            economic_cycles = []
            for year in national_data['Year']:
                if year in [1988, 2003, 2008, 2015, 2021]:  # Challenges
                    economic_cycles.append(-1)
                elif year in [1992, 2000, 2011, 2016, 2022]:  # Recovery
                    economic_cycles.append(1)
                else:
                    economic_cycles.append(0)
            
            fig.add_trace(
                go.Scatter(
                    x=national_data['Year'],
                    y=economic_cycles,
                    mode='lines+markers',
                    name='Economic Cycle',
                    line=dict(color='orange', width=3),
                    marker=dict(size=6),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Cycle:</b> %{y}<extra></extra>'
                ),
                row=row, col=col,
                secondary_y=False
            )
            
            # Add agriculture values on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=national_data['Year'],
                    y=national_data['Agriculture_Value_Added'],
                    mode='lines',
                    name='Agriculture Value',
                    line=dict(color=self.colors['national'], width=2),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Value:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col,
                secondary_y=True
            )
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Economic Cycle", secondary_y=False, row=row, col=col)
            fig.update_yaxes(title_text="Agriculture Value (%)", secondary_y=True, row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Economic Cycle Data")
    
    def _add_uncertainty_trend_analysis(self, fig, forecasts, row, col):
        """Add uncertainty trend analysis"""
        if 'national' in forecasts:
            forecast_data = forecasts['national']
            years = [item['year'] for item in forecast_data]
            confidences = [item['confidence'] for item in forecast_data]
            means = [item['mean'] for item in forecast_data]
            
            # Calculate uncertainty metrics
            relative_uncertainty = [(conf / mean) * 100 if mean != 0 else 0 for conf, mean in zip(confidences, means)]
            
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=relative_uncertainty,
                    mode='lines+markers',
                    name='Relative Uncertainty',
                    line=dict(color='red', width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Year:</b> %{x}<br><b>Uncertainty:</b> %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add trend line
            if len(years) > 1:
                z = np.polyfit(range(len(years)), relative_uncertainty, 1)
                trend_line = np.poly1d(z)(range(len(years)))
                
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=trend_line,
                        mode='lines',
                        name='Uncertainty Trend',
                        line=dict(color='blue', width=2, dash='dash'),
                        hovertemplate='<b>Trend</b><extra></extra>'
                    ),
                    row=row, col=col
                )
            
            fig.update_xaxes(title_text="Year", row=row, col=col)
            fig.update_yaxes(title_text="Relative Uncertainty (%)", row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Uncertainty Trend Data")
    
    def _add_recommendation_summary(self, fig, forecasts, row, col):
        """Add recommendation summary as a table"""
        recommendations = self._generate_recommendation_data(forecasts)
        
        if recommendations:
            # Create a table-like visualization
            fig.add_trace(
                go.Scatter(
                    x=[0] * len(recommendations),
                    y=list(range(len(recommendations))),
                    mode='markers+text',
                    marker=dict(size=0.1, color='white'),
                    text=recommendations,
                    textposition="middle left",
                    hoverinfo='none',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Set appropriate axis ranges
            fig.update_xaxes(showticklabels=False, range=[-0.1, 1], row=row, col=col)
            fig.update_yaxes(showticklabels=False, range=[-1, len(recommendations)], row=row, col=col)
        else:
            self._add_no_data_message(fig, row, col, "Recommendations")
    
    def _generate_recommendation_data(self, forecasts):
        """Generate recommendation data for display"""
        recommendations = []
        
        if 'national' in forecasts:
            national_forecast = forecasts['national']
            if national_forecast:
                current_value = national_forecast[0]['mean']
                future_value = national_forecast[-1]['mean']
                growth_rate = ((future_value - current_value) / current_value * 100) if current_value > 0 else 0
                
                # Generate recommendations based on growth rate
                if growth_rate > 3:
                    outlook = "🟢 HIGHLY POSITIVE"
                    main_rec = "Expand cultivation & invest in high-value crops"
                elif growth_rate > 1:
                    outlook = "🟡 POSITIVE"
                    main_rec = "Maintain operations with efficiency focus"
                elif growth_rate > -1:
                    outlook = "🟠 STABLE"
                    main_rec = "Focus on risk management & cost optimization"
                else:
                    outlook = "🔴 CHALLENGING"
                    main_rec = "Implement defensive strategies & diversify"
                
                recommendations.extend([
                    f"📊 Market Outlook: {outlook}",
                    f"📈 Growth Rate: {growth_rate:+.1f}%",
                    f"💡 Key Action: {main_rec}",
                    "",
                    "🌾 Crop Strategies:",
                    "• Rice: Improve water management",
                    "• Pulses: Expand production",
                    "• Sesame: Export focus",
                    "• Maize: Quality improvement",
                    "",
                    "🛡️ Risk Management:",
                    "• Diversify crop portfolio",
                    "• Climate resilience",
                    "• Market monitoring"
                ])
        
        return recommendations
    
    def _add_no_data_message(self, fig, row, col, message):
        """Add a 'no data' message to a subplot"""
        fig.add_trace(
            go.Scatter(
                x=[0.5],
                y=[0.5],
                mode='text',
                text=[f'No {message} Available'],
                textfont=dict(size=14, color='gray'),
                showlegend=False,
                hoverinfo='none'
            ),
            row=row, col=col
        )
        fig.update_xaxes(showticklabels=False, range=[0, 1], row=row, col=col)
        fig.update_yaxes(showticklabels=False, range=[0, 1], row=row, col=col)
    
    def create_simple_dashboard(self, historical_data, forecasts, output_file="simple_dashboard.html"):
        """Create a simplified dashboard that's guaranteed to work"""
        print("Creating simplified dashboard...")
        
        # Create a simpler 2x2 dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'National Agriculture Trend & Forecast',
                'Growth Rate Analysis',
                'Regional Forecast Comparison',
                'Data Completeness Overview'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Add basic visualizations
        self._add_simple_national_trend(fig, historical_data, forecasts, row=1, col=1)
        self._add_simple_growth_analysis(fig, forecasts, row=1, col=2)
        self._add_simple_regional_comparison(fig, forecasts, row=2, col=1)
        self._add_simple_data_completeness(fig, historical_data, row=2, col=2)
        
        fig.update_layout(
            title_text="Myanmar Agricultural Forecasting - Simplified Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        fig.write_html(output_file)
        print(f"✅ Simple dashboard saved as {output_file}")
        return fig
    
    def _add_simple_national_trend(self, fig, historical_data, forecasts, row, col):
        """Add simplified national trend"""
        if historical_data and 'national' in historical_data and forecasts and 'national' in forecasts:
            national_data = historical_data['national']
            national_forecast = forecasts['national']
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=national_data['Year'],
                    y=national_data['Agriculture_Value_Added'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=3)
                ),
                row=row, col=col
            )
            
            # Forecast data
            forecast_years = [item['year'] for item in national_forecast]
            forecast_means = [item['mean'] for item in national_forecast]
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_years,
                    y=forecast_means,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3, dash='dash')
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Year", row=row, col=col)
        fig.update_yaxes(title_text="Agriculture Value Added (% of GDP)", row=row, col=col)
    
    def _add_simple_growth_analysis(self, fig, forecasts, row, col):
        """Add simplified growth analysis"""
        if 'national' in forecasts:
            forecast_data = forecasts['national']
            years = [item['year'] for item in forecast_data]
            means = [item['mean'] for item in forecast_data]
            
            growth_rates = []
            for i in range(1, len(means)):
                growth = ((means[i] - means[i-1]) / means[i-1]) * 100 if means[i-1] != 0 else 0
                growth_rates.append(growth)
            
            fig.add_trace(
                go.Bar(
                    x=years[1:],
                    y=growth_rates,
                    name='Growth Rate',
                    marker_color=['green' if x > 0 else 'red' for x in growth_rates]
                ),
                row=row, col=col
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black", row=row, col=col)
        
        fig.update_xaxes(title_text="Year", row=row, col=col)
        fig.update_yaxes(title_text="Growth Rate (%)", row=row, col=col)
    
    def _add_simple_regional_comparison(self, fig, forecasts, row, col):
        """Add simplified regional comparison"""
        regional_forecasts = {k: v for k, v in forecasts.items() if k != 'national'}
        
        if regional_forecasts:
            for region, forecast_data in regional_forecasts.items():
                years = [item['year'] for item in forecast_data]
                means = [item['mean'] for item in forecast_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=means,
                        mode='lines',
                        name=region
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(title_text="Year", row=row, col=col)
        fig.update_yaxes(title_text="Agriculture Value", row=row, col=col)
    
    def _add_simple_data_completeness(self, fig, historical_data, row, col):
        """Add simplified data completeness"""
        datasets = ['National', 'Union', 'State_Region', 'Township']
        completeness = []
        
        for key in ['national', 'union', 'state_region', 'township']:
            if historical_data.get(key) is not None:
                df = historical_data[key]
                comp = min(100, df.count().sum() / (len(df.columns) * len(df)) * 100)
                completeness.append(comp)
            else:
                completeness.append(0)
        
        fig.add_trace(
            go.Bar(
                x=datasets,
                y=completeness,
                name='Data Completeness',
                marker_color=['green' if x > 80 else 'orange' if x > 50 else 'red' for x in completeness]
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Dataset", row=row, col=col)
        fig.update_yaxes(title_text="Completeness (%)", row=row, col=col, range=[0, 100])
    
    def generate_enhanced_farmer_report(self, forecasts, output_file="enhanced_farmer_recommendations.html"):
        """Generate enhanced farmer-friendly HTML report"""
        
        # Calculate key metrics
        national_forecast = forecasts.get('national', [])
        if national_forecast:
            current_value = national_forecast[0]['mean'] if national_forecast else 0
            future_value = national_forecast[-1]['mean'] if national_forecast else 0
            growth_rate = ((future_value - current_value) / current_value * 100) if current_value > 0 else 0
            
            # Calculate uncertainty
            avg_confidence = np.mean([item['confidence'] for item in national_forecast])
            relative_uncertainty = (avg_confidence / current_value * 100) if current_value > 0 else 0
        else:
            current_value = future_value = growth_rate = relative_uncertainty = 0
        
        # Generate recommendations based on metrics
        outlook, color, risk_level, main_recommendation = self._calculate_recommendation_level(growth_rate, relative_uncertainty)
        
        # Feature importance insights
        feature_insights = self._get_feature_insights()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Myanmar Agriculture Forecast Report</title>
            <style>
                :root {{
                    --primary-color: #2E8B57;
                    --secondary-color: #4682B4;
                    --positive-color: #00AA00;
                    --negative-color: #FF4444;
                    --neutral-color: #FFAA00;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    color: #333;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                    color: white;
                    padding: 30px;
                    text-align: center;
                    border-radius: 15px;
                    margin-bottom: 30px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    margin: 0;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 1.2em;
                    opacity: 0.9;
                }}
                
                .dashboard {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                }}
                
                .card h2 {{
                    color: var(--primary-color);
                    margin-top: 0;
                    border-bottom: 2px solid #f0f0f0;
                    padding-bottom: 10px;
                }}
                
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                
                .metric {{
                    text-align: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid var(--primary-color);
                }}
                
                .metric-value {{
                    font-size: 1.8em;
                    font-weight: bold;
                    margin: 5px 0;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: #666;
                }}
                
                .outlook-{color} {{
                    background: {'#e8f5e8' if color == 'positive' else '#fff3cd' if color == 'neutral' else '#ffe6e6'};
                    border-left: 4px solid {'var(--positive-color)' if color == 'positive' else 'var(--neutral-color)' if color == 'neutral' else 'var(--negative-color)'};
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                
                .recommendation-list {{
                    list-style: none;
                    padding: 0;
                }}
                
                .recommendation-list li {{
                    padding: 10px 0;
                    border-bottom: 1px solid #f0f0f0;
                    display: flex;
                    align-items: center;
                }}
                
                .recommendation-list li:before {{
                    content: "✓";
                    color: var(--positive-color);
                    font-weight: bold;
                    margin-right: 10px;
                }}
                
                .feature-insights {{
                    background: #e3f2fd;
                    padding: 20px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                
                .feature-item {{
                    margin: 10px 0;
                    padding: 10px;
                    background: white;
                    border-radius: 6px;
                    border-left: 4px solid var(--secondary-color);
                }}
                
                @media (max-width: 768px) {{
                    .dashboard {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .header h1 {{
                        font-size: 2em;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🌾 Myanmar Agricultural Forecast Report</h1>
                    <p>Comprehensive Analysis & Recommendations • Generated on {datetime.now().strftime('%B %d, %Y')}</p>
                </div>
                
                <div class="dashboard">
                    <div class="card">
                        <h2>📊 Executive Summary</h2>
                        <div class="outlook-{color}">
                            <h3 style="margin: 0; color: {'var(--positive-color)' if color == 'positive' else 'var(--neutral-color)' if color == 'neutral' else 'var(--negative-color)'};">{outlook}</h3>
                            <p style="margin: 10px 0 0 0; font-weight: bold;">{main_recommendation}</p>
                        </div>
                        
                        <div class="metric-grid">
                            <div class="metric">
                                <div class="metric-label">Current Value</div>
                                <div class="metric-value" style="color: var(--primary-color);">{current_value:.1f}%</div>
                                <div class="metric-label">of GDP</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">3-Year Projection</div>
                                <div class="metric-value" style="color: var(--secondary-color);">{future_value:.1f}%</div>
                                <div class="metric-label">of GDP</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Growth Rate</div>
                                <div class="metric-value" style="color: {'var(--positive-color)' if growth_rate > 0 else 'var(--negative-color)'};">{growth_rate:+.1f}%</div>
                                <div class="metric-label">annual</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Risk Level</div>
                                <div class="metric-value" style="color: {'var(--positive-color)' if risk_level == 'Low' else 'var(--neutral-color)' if risk_level == 'Medium' else 'var(--negative-color)'};">{risk_level}</div>
                                <div class="metric-label">uncertainty</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>💡 Strategic Recommendations</h2>
                        <ul class="recommendation-list">
                            <li>Review planting plans based on forecast outlook</li>
                            <li>Implement recommended risk management strategies</li>
                            <li>Diversify crop portfolio to spread risk</li>
                            <li>Focus on climate-resilient agricultural practices</li>
                            <li>Explore market diversification opportunities</li>
                            <li>Invest in water conservation technologies</li>
                            <li>Monitor agricultural bulletins regularly</li>
                            <li>Consider cooperative farming models</li>
                        </ul>
                    </div>
                </div>
                
                <div class="dashboard">
                    <div class="card">
                        <h2>🌱 Crop-Specific Guidance</h2>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                            <div style="background: #e8f5e8; padding: 15px; border-radius: 8px;">
                                <h4 style="margin: 0 0 10px 0; color: var(--positive-color);">✅ Favorable Crops</h4>
                                <ul style="margin: 0; padding-left: 20px;">
                                    <li>Rice (improved varieties)</li>
                                    <li>Pulses (expanding demand)</li>
                                    <li>Sesame (export potential)</li>
                                    <li>Green gram (market stability)</li>
                                </ul>
                            </div>
                            <div style="background: #fff3cd; padding: 15px; border-radius: 8px;">
                                <h4 style="margin: 0 0 10px 0; color: var(--neutral-color);">⚠️ Monitor Closely</h4>
                                <ul style="margin: 0; padding-left: 20px;">
                                    <li>Maize (quality focus)</li>
                                    <li>Groundnut (market prices)</li>
                                    <li>Sugarcane (contract farming)</li>
                                    <li>Vegetables (local demand)</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>🔍 Key Success Factors</h2>
                        <div class="feature-insights">
                            {feature_insights}
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🛡️ Risk Management Framework</h2>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                        <div>
                            <h4>🌦️ Climate Risks</h4>
                            <ul>
                                <li>Implement drought-resistant varieties</li>
                                <li>Practice water conservation</li>
                                <li>Diversify planting schedules</li>
                                <li>Use weather forecasting tools</li>
                            </ul>
                        </div>
                        <div>
                            <h4>📈 Market Risks</h4>
                            <ul>
                                <li>Monitor price fluctuations</li>
                                <li>Explore contract farming</li>
                                <li>Diversify market channels</li>
                                <li>Join farmer cooperatives</li>
                            </ul>
                        </div>
                        <div>
                            <h4>💼 Financial Risks</h4>
                            <ul>
                                <li>Maintain cost records</li>
                                <li>Explore crop insurance</li>
                                <li>Diversify income sources</li>
                                <li>Plan for price volatility</li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <footer style="text-align: center; margin-top: 40px; padding: 20px; background: white; border-radius: 12px;">
                    <p style="margin: 0; color: #666;">Generated by Myanmar Agricultural Forecasting System</p>
                    <p style="margin: 5px 0 0 0; color: #888;">FTL Myanmar Machine Learning Bootcamp - Group 4</p>
                    <p style="margin: 5px 0 0 0; color: #888;">For educational and research purposes</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Enhanced farmer report saved as {output_file}")
        return html_content
    
    def _calculate_recommendation_level(self, growth_rate, uncertainty):
        """Calculate recommendation level based on growth rate and uncertainty"""
        if growth_rate > 3 and uncertainty < 10:
            return "HIGHLY POSITIVE OUTLOOK", "positive", "Low", "Consider expanding cultivation areas and investing in high-value crops"
        elif growth_rate > 1 and uncertainty < 15:
            return "POSITIVE OUTLOOK", "positive", "Medium-Low", "Maintain current operations with focus on efficiency improvements"
        elif growth_rate > -1:
            return "STABLE OUTLOOK", "neutral", "Medium", "Focus on risk management and cost optimization"
        else:
            return "CHALLENGING OUTLOOK", "negative", "High", "Implement defensive strategies and diversify income sources"
    
    def _get_feature_insights(self):
        """Get feature importance insights for the report"""
        insights = ""
        
        if hasattr(self.forecaster, 'feature_importance') and 'national' in self.forecaster.feature_importance:
            top_features = self.forecaster.feature_importance['national'].head(3)
            
            for _, row in top_features.iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if 'Lag' in feature:
                    insight = f"Historical performance patterns strongly influence predictions"
                elif 'Rolling' in feature:
                    insight = f"Recent trend momentum is a key predictor"
                elif 'Economic' in feature:
                    insight = f"Economic cycles significantly impact outcomes"
                elif 'Year' in feature:
                    insight = f"Long-term time trends drive forecasts"
                else:
                    insight = f"Statistical patterns guide predictions"
                
                insights += f'<div class="feature-item"><strong>{feature}</strong> ({importance:.1%} impact)<br><small>{insight}</small></div>'
        
        return insights if insights else '<div class="feature-item">Feature importance data not available</div>'

def main():
    """Main function to generate enhanced dashboard and reports"""
    print("🚀 Generating Enhanced Agricultural Forecasting Dashboard...")
    print("=" * 60)
    
    try:
        # Import and initialize the enhanced forecaster
        from myanmar_agricultural_forecasting_system import MyanmarAgriculturalForecaster
        
        # Initialize components
        forecaster = MyanmarAgriculturalForecaster()
        dashboard_gen = EnhancedDashboardGenerator(forecaster)
        
        # Load data and generate forecasts
        print("📊 Loading data and generating forecasts...")
        integrated_data = forecaster.load_and_integrate_data()
        
        if integrated_data is None:
            print("❌ Failed to load data. Please check your data files.")
            return
        
        # Train models and generate forecasts
        forecaster.train_enhanced_models(integrated_data)
        forecasts = forecaster.generate_enhanced_forecasts(years_ahead=5)
        
        # Try to create comprehensive dashboard first
        try:
            print("🎨 Creating comprehensive interactive dashboard...")
            dashboard_fig = dashboard_gen.create_comprehensive_dashboard(
                integrated_data, forecasts, "enhanced_agriculture_dashboard.html"
            )
        except Exception as e:
            print(f"⚠️ Comprehensive dashboard failed, creating simplified version: {e}")
            dashboard_fig = dashboard_gen.create_simple_dashboard(
                integrated_data, forecasts, "agriculture_dashboard.html"
            )
        
        # Generate enhanced farmer report
        print("📝 Generating farmer recommendations...")
        farmer_report = dashboard_gen.generate_enhanced_farmer_report(
            forecasts, "farmer_recommendations.html"
        )
        
        print("\n" + "=" * 60)
        print("✅ DASHBOARD GENERATION COMPLETE")
        print("=" * 60)
        print("\n📁 Generated Files:")
        print("   • agriculture_dashboard.html - Interactive visualization dashboard")
        print("   • farmer_recommendations.html - Comprehensive farmer report")
        
        print("\n🎯 Next Steps:")
        print("   1. Open the HTML files in your web browser")
        print("   2. Share the farmer report with agricultural stakeholders")
        print("   3. Use the dashboard for policy decisions and planning")
        
        # Display key insights
        if forecasts and 'national' in forecasts:
            national_forecast = forecasts['national']
            if national_forecast:
                current = national_forecast[0]['mean']
                future = national_forecast[-1]['mean']
                growth = ((future - current) / current * 100) if current > 0 else 0
                
                print(f"\n📈 Key Insight: Agriculture GDP projected to {'grow' if growth > 0 else 'decline'} by {abs(growth):.1f}% over 5 years")
        
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        print("💡 Make sure myanmar_agricultural_forecasting_system.py is in the same directory")

if __name__ == "__main__":
    main()