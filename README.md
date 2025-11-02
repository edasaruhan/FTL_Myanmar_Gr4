MYANMAR AGRICULTURAL FORECASTING SYSTEM
Version 1.0
FTL Myanmar Machine Learning Bootcamp - Group 4


==================================================
APPENDIX: FILE STRUCTURE
==================================================

Project Directory:
/
├── generate.py                 # Dashboard and report generation
├── myanmar_agricultural_forecasting_system.py  # Main system
├── user_manual.txt            # This manual
├── requirements.txt           # Python dependencies
├── enhanced_agriculture_dashboard.html # Generated dashboard
├── farmer_recommendations.html # Generated farmer report
└── data/                      # Data directory (optional)
    ├── API_NV.AGR.TOTL.ZS_DS2_en_excel_v2_128618.xls
    └── MIMU_BaselineData_Agriculture_Countrywide_5Mar2025.xlsx

Generated Files:
- enhanced_agriculture_dashboard.html: Interactive web dashboard
- farmer_recommendations.html: HTML report for farmers
- Console output: Numerical forecasts and recommendations


==================================================
1. SYSTEM OVERVIEW
==================================================

The Myanmar Agricultural Forecasting System is a machine learning-based tool that predicts agricultural trends using World Bank and MIMU data. The system helps farmers, policymakers, and agricultural stakeholders make data-driven decisions.

Key Features:
- National and regional agricultural forecasting
- XGBoost machine learning models
- Interactive visualizations
- Farmer-friendly recommendations
- Climate resilience insights


==================================================
2. QUICK START
==================================================

For immediate use:

1. Place data files in the project directory
2. Run the main forecasting system:
   python myanmar_agricultural_forecasting_system.py

3. Generate dashboard and reports:
   python generate.py

Expected Output:
- agriculture_dashboard.html (Interactive dashboard)
- farmer_recommendations.html (Farmer report)
- Console output with forecasts and recommendations


License:
This system is developed for educational and research purposes as part of the FTL Myanmar Machine Learning Bootcamp.


==================================================
LAST UPDATED: November 2025
==================================================