library(dplyr)
library(lubridate)
library(ggvis)
library(ggplot2)

df <- read.csv("cleandata.csv")
str(df)
df$ArrDelay15 <- factor(df$ArrDelay15)
df$DepDelay15 <- factor(df$DepDelay15)
df$Month <- match(df$Month,month.abb)
df$Delay15 <- factor(df$Delay15)

airports <-read.csv("airports.csv")

df3 <- df %>% select(Month,DayofMonth,DayOfWeek,
                     UniqueCarrier,
                     ArrDelay,DepDelay,Delay15,
                     Origin,Dest,Cancelled,Diverted,
                     CarrierDelay,WeatherDelay,NASDelay,SecurityDelay,LateAircraftDelay,
                     lat_origin,long_origin,lat_dest,long_dest)

df3 <- subset(df3,Delay15==1) ##STOPPED HERE. Compare df3 to delay_FL and see if we can use df3 instaed

# Key, value pairs of Airline carriers and two letter codes
airline_name = c('Endeavor Air' = '9E', 'American Airlines' = 'AA','9 Air Co' = 'AQ', 'Alaska Airlines' = 'AS',
             'Jetblue Airways' = 'B6', 'Cobaltair' = 'CO', 'Delta Airlines' = 'DL', 'ExpressJet Airlines' = 'EV',
             'Frontier Airlines' = 'F9', 'AirTran Airways' = 'FL', 'Hawaiian Airlines' = 'HA', 'Envoy Air' = 'MQ',
             'Northwest Airline' = 'NW', 'Comair' = 'OH', 'United Airlines' = 'UA', 'US Airways' = 'US', 
             'Southwest Airlines' = 'WN', 'Mesa Air Group' = 'YV')


df1 <- tbl_df(df)
df1

airlines <- group_by(df1,Description)
airlines

df2 <- summarise(airlines, total_delay=sum(ArrDelay15==1))
df2

# Trending Monthly, Daily, Hourly flights
flights_daily <- group_by(df1,Description,DayOfWeek) %>%summarise('total_flights'=n())
flights_monthly <- group_by(df1,Description,Month) %>%summarise('total_flights'=n())
flights_hourly <- group_by(df1,UniqueCarrier,Description,HOUR) %>%summarise('total_flights'=n())

#Testing
kk = flights_hourly %>% filter(UniqueCarrier == 'AA')
ggplot(kk, aes(x=HOUR,y=total_flights,fill=HOUR)) + geom_bar(width=0.9,stat = 'identity') +  labs(y='# of flights', x='Hours') + scale_x_continuous(breaks=0:23,
                                                                                                                                           labels=c(24,1,2,3,4,5,6,7,8,9,10,
                                                                                                                                                    11,12,13,14,15,16,17,18,
                                                                                                                                                19,20,21,22,23)) + theme(axis.text.x=element_text(angle=0,vjust=0.5))
ggvis(kk,~HOUR,~total_flights) %>% layer_bars()
ggvis(flights_year,~Description,~total_flights) %>%layer_bars() %>%  add_axis("x", properties = axis_props(labels = list(angle = 45, align = "left", baseline = "middle")))


saveRDS(flights_daily,"flights_daily.RDS")
saveRDS(flights_monthly,"flights_monthly.RDS")
saveRDS(flights_hourly,"flights_hourly.RDS")


#of total flights in a year (count, hours(in minutes), distance(in miles))
flights_year <- group_by(df1,Description) %>%summarise('total_flights'=n(),'total_hours'=sum(HOUR),'total_dist'=sum(Distance)) %>% arrange(desc(total_flights))
flights_year

#Delays only
delays <- df2[df2$WeatherDelay!='NA',]
delays$Month <- match(delays$Month,month.abb)
delayss <- group_by(df1,Origin,Dest,Month) %>% summarise("total_num_delays"=sum(ArrDelay15==1),"total_flights"=n())
delayss$Month <- match(delayss$Month,month.abb)
delayss <- filter(delayss ,Month >= 1 & Month <= 6& Origin == "ABE")
delayss <- group_by(delayss,Origin,Dest) %>%summarise("total_num_delays"=sum(total_num_delays),"total_flights"=sum(total_flights))

delayss1 <- filter(delayss ,Month >= 1 & Month <= 12 & Origin == "ABE")

#total # of flights by route
routes <- df1 %>% group_by(Origin,Dest) %>% summarise(total_flights=n())

saveRDS(df3,"delaytry.RDS")
saveRDS(routes, "routes.RDS")

#For each airline, which airport are they delayed most 
df$HOUR <- hms(as.character(df$DepTime))
flights_hourly <- group_by(df1,HOUR) %>%summarise('total_flights'=n())
flights_hourly
tail(flights_hourly,10)