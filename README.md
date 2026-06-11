# Time series analysis of PM2.5 air quality in Nairobi

### Project description

This project analyses PM2.5 (fine particulate matter) air quality readings from Nairobi, Kenya, using low-cost sensor data from the sensors.AFRICA network — a Code for Africa initiative with deployments across multiple African cities. The goal is to model the temporal structure of PM2.5 concentrations and build forecasting models.

### Problem statement

Can we build reliable time series models to forecast hourly PM2.5 levels in Nairobi, and what do these models reveal about the temporal dynamics of air quality in the city? The EPA 24-hour standard for PM2.5 is 35 µg/m³. Understanding the autocorrelation structure of these readings enables short-term forecasting and informs pollution response decisions.

### Source

AFRICA Air Quality Archive, downloaded from openAfrica (open.africa). Monthly CSV files with semicolon delimiters containing PM, temperature, and humidity readings from PMS5003 and DHT22 sensors.


### Methodology

 - Exploratory data analysis


 - Stationarity testing


 - ACF and PACF analysis


 - Model fitting and evaluation