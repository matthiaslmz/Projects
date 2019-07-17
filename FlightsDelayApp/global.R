library(leaflet)
library(sp)


flights_Hourly=readRDS('flights_hourly.RDS')
flights_Daily=readRDS('flights_daily.RDS')
flights_Monthly=readRDS('flights_monthly.RDS')
delay_HM <- readRDS('delay_heatmap.RDS')
delay_AD <- readRDS('delay_arrdel.RDS')
delay_ADM <- readRDS('delay_arrdelmins.RDS')
delay_avgDM <- readRDS('delay_avgdelmins.RDS')
delay_TD <- readRDS('delay_typedelays.RDS')
#delay_FL <- readRDS('delays.RDS')
delay_FL <- readRDS('delaytry.RDS')
routes <- readRDS("routes.RDS")
airports <- read.csv("airports.csv")
airport_codes <- levels(delay_FL$Origin)


# Key, value pairs of Airline carriers and two letter codes
airline_name = c('Endeavor Air' = '9E', 'American Airlines' = 'AA','9 Air Co' = 'AQ', 'Alaska Airlines' = 'AS',
                 'Jetblue Airways' = 'B6', 'Cobaltair' = 'CO', 'Delta Airlines' = 'DL', 'ExpressJet Airlines' = 'EV',
                 'Frontier Airlines' = 'F9', 'AirTran Airways' = 'FL', 'Hawaiian Airlines' = 'HA', 'Envoy Air' = 'MQ',
                 'Northwest Airline' = 'NW', 'Comair' = 'OH', 'United Airlines' = 'UA', 'US Airways' = 'US', 
                 'Southwest Airlines' = 'WN', 'Mesa Air Group' = 'YV')

#Leaflet map

map <- leaflet(width=900,height=650) %>% setView(lng=-95.72,lat=37.13,zoom=4) %>% 
  addTiles('http://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
           options = providerTileOptions(noWrap = TRUE))
  


