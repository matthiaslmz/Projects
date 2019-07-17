library(shiny)
library(shinydashboard)
library(ggplot2)
library(viridis)
library(reshape2)
library(RColorBrewer)
library(leaflet)
library(geosphere)
library(DT)
library(rsconnect)

shinyServer(function(input, output){

#### REACTIVE FUNCTONS ####
  
hour_gen = reactive({
  hours = flights_Hourly
  hours = hours %>% filter(UniqueCarrier == airline_name[input$airlines])
})

daily_gen = reactive({
  days = flights_Daily
  days = days %>% filter(UniqueCarrier == airline_name[input$airlines])
})

monthly_gen = reactive({
  months = flights_Monthly
  months = months %>% filter(UniqueCarrier == airline_name[input$airlines])
})


delay_gen = reactive({
  heat = delay_HM
  heat = delay_HM %>% filter(UniqueCarrier == airline_name[input$airlines])
})

arrdelay_gen = reactive({
  arrd = delay_AD
  arrd = delay_AD %>% filter(UniqueCarrier == airline_name[input$airlines])
  arrd = head(arrd,10)
})

arrdelaymins_gen = reactive({
  arrd1 = delay_ADM
  arrd1 = delay_ADM %>% filter(UniqueCarrier == airline_name[input$airlines])
})

avgdelaymins_gen = reactive({
  avg = delay_avgDM
  avg = delay_avgDM %>% filter(UniqueCarrier == airline_name[input$airlines])
})

typedelays_gen = reactive({
  td = delay_TD
  td = delay_TD %>% filter(UniqueCarrier == airline_name[input$airlines])
})

#For map tab, filtering for delays for data table
delay_filter = reactive({
  
  routes_delay = delay_FL %>% filter(Month >= input$months[1] & 
                                                Month <= input$months[2] & 
                                                Origin == input$airport)
  # grouping delays by Origin and Destination and inner join with routes data for total_flights column
  routes_delay = routes_delay %>% group_by(Origin, Dest) %>% 
    inner_join(routes, c('Origin','Dest')) %>% 
    summarise(total_delays = n()) %>% inner_join(routes, by = c('Origin','Dest'))
  
  routes_delay['Percent'] = round(100*routes_delay$total_delays/routes_delay$total_flights, 2)
  
  routes_delay
})

# Get top 10 delays for plotting great circle lines
top10delay_gen = reactive({
  routes_delay = delay_filter()
  
 
  sort_choice = switch(input$sort_choice,
                       "Total Delays" = "-total_delays",
                       "Percentage" = "-Percent")
  
  # Sort delay routes by descending order
  delays1 = routes_delay %>% arrange_(sort_choice)
  
  #Subset to get top 10 delays
  end_ind = min(nrow(delays1), 10)
  delays_sub = delays1[1:end_ind,] %>% inner_join(airports, by = c('Dest' = 'iata'))
  delays_sub$Percent = round(delays_sub$Percent,2)
  delays_sub
  
})


#### OUTPUT FUNCTIONS ####


output$selected_var <-renderText({
  input$airlines
})

output$totalflights <-renderText({
  tt = monthly_gen
  s = format(round(as.numeric(sum(tt()$total_flights))),big.mark=',')
  s
})

output$totaldist <-renderText({
  tt = monthly_gen
  s = format(round(as.numeric(sum(tt()$total_distance))),big.mark=',')
  s
})

output$selected_var1 <-renderText({
  input$airlines
})

output$totaldelaymins <-renderText({
  dd = arrdelaymins_gen
  ds = format(round(as.numeric(sum(dd()$total_delaymins))),big.mark=',')
  ds
})

output$avgdelaymins <-renderText({
  dd = arrdelaymins_gen
  ds = format(round(as.numeric(dd()$total_delaymins/dd()$total_flightsdelayed),2),big.mark=",")
  ds
})

output$table <- renderDataTable({
  cols <- c("Origin","Dest","total_delays","total_flights","Percent")
  delays_sub <- delay_filter()
  datatable(delays_sub[cols], rownames=FALSE) %>% formatStyle(cols, fontWeight='bold',color= "black")
})

output$omap = renderLeaflet({
  delays_sub = top10delay_gen()
  
  #Starting coordinates
  lat_start = airports$lat[airports$iata==input$airport]
  long_start = airports$long[airports$iata==input$airport]
  
  #Ending coordinates
  lat_end = delays_sub$lat
  long_end = delays_sub$long
  
  tmap = map
  
  # Adding great circle lines 
  for (i in 1:length(delays_sub$Dest)){
  
    
    detail <- paste('Destination: ', airports[airports$iata == delays_sub$Dest[i],'airport'], "<br>" ,
                   'Total Delays = ', delays_sub$total_delays[i], '<br>',
                   'Total Flights = ', routes$total_flights[routes$Origin == input$airport & 
                                                              routes$Dest == delays_sub$Dest[i]],'<br>',
                   'Percentage = ', delays_sub$Percent[i], '%', sep = '')

    # Instantiating great circle lines
    inter <- gcIntermediate(c(long_start, lat_start), c(long_end[i], lat_end[i]), n=200, addStartEnd=TRUE)
    tmap = tmap %>% addPolylines(data = inter, weight = 2.5, color = "red", opacity = 0.9 ,popup = detail)
  }
  tmap

})

output$barcharthour = renderPlot({
  h = hour_gen
  title_ = paste('Hourly Trend', " on ", input$airlines, sep='')
  print(ggplot(data=h(), aes(x=HOUR,y=total_flights,fill=HOUR)) + geom_bar(width=0.9,stat = 'identity') +  
          labs(y='# of flights', x='Hours',title=title_) + scale_x_continuous(breaks=0:23, labels=c(24,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)) 
        + theme(axis.text.x=element_text(angle=0,vjust=0.5),legend.position="none",plot.title=element_text(hjust=0.5)))
})
output$barchartday = renderPlot({
  d = daily_gen
  title_ = paste('Daily Trend', " on ", input$airlines, sep='')
  print(ggplot(d(), aes(x=DayOfWeek,y=total_flights,fill=DayOfWeek)) + geom_bar(width=0.9,stat = 'identity') +  labs(y='# of Flights', x='Days',title=title_) + theme(axis.text.x=element_text(angle=90,vjust=0.5),legend.position="none",plot.title=element_text(hjust=0.5)) + scale_fill_brewer(palette='YlGnBu',direction=-1))
})
output$barchartmonth = renderPlot({
  m = monthly_gen
  title_ = paste('Monthly Trend', " on ", input$airlines, sep='')
  print(ggplot(m(), aes(x=Month,y=total_flights,fill=Month)) + geom_bar(width=0.9,stat = 'identity') +  labs(y='# of Flights', x='Months',title=title_) + theme(legend.position="none",plot.title=element_text(hjust=0.5)))
})

output$delayheat = renderPlot({
  d = delay_gen
  title_ = paste('Delay Heatmap', " on ", input$airlines, " (COUNT)", sep='')
  print(ggplot(d(),aes(x=Month,y=DayofMonth)) + geom_tile(aes(fill=total_delay)) + scale_fill_viridis(option='magma',direction=-1) + geom_text(aes(label=total_delay),color="turquoise3",size=3) +  labs(y='#Day', x='Month',title=title_)+ theme(plot.title=element_text(hjust=0.5)))
})

output$arrdelay = renderPlot({
  ad = arrdelay_gen
  title_ = paste('Top 10 Airport (Origin) Delays', " on ", input$airlines, sep='')
  ggplot(ad(),aes(x=airport_origin,y=total_delaymins)) + geom_point(size=5,color='blue',alpha=0.9) + 
    geom_segment(aes(x=airport_origin,xend=airport_origin,y=0,yend=total_delaymins),color="turquoise4",size=1,alpha=0.6) + 
    labs(title=title_,y='Delays (minutes)',x='Airports') +   theme_light() + theme(plot.title=element_text(hjust=0.5), panel.grid.major.y=element_blank(),panel.border=element_blank(),legend.position="none") + 
    coord_flip()
})
output$typedelays = renderPlot({
  dfm = typedelays_gen
  title_ = paste('Type of Delays', " on ", input$airlines, sep='')
  ggplot(dfm(),aes(x=variable,y=value,fill=variable)) + geom_bar(stat='identity') +  labs(y='Number of Delays (minutes)', x='Types of Delay ',title=title_) +  scale_fill_brewer(palette='Dark2',direction=-1) + theme(axis.text.x=element_text(angle=60,vjust=1,hjust=0.8),plot.title=element_text(hjust=0.5))
})
   
})
