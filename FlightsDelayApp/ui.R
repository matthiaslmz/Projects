library(shiny)
library(shinydashboard)
library(ggplot2)
library(dplyr)
library(DT)
library(leaflet)
library(geosphere)

ui <- dashboardPage(
  dashboardHeader(title = "2008 Airline Flight Statistics",titleWidth = 350),
  dashboardSidebar(
    width=350,
    sidebarMenu(
      menuItem("Introduction", tabName = "Intro", icon = icon("dashboard")),
      menuItem("Summary", tabName = "Summary", icon = icon("line-chart")),
      menuItem("Delays", tabName = "Delays", icon = icon("clock-o")),
      menuItem("Map", tabName = "Map", icon = icon("map")),
      menuSubItem(icon=NULL,selectInput("airlines",label = "Airline Carrier",choices = c(names(airline_name)), selected = 'American Airlines'))
    )
  ),
  dashboardBody(
    tags$head(
      tags$link(rel="stylesheet",type="text/css",href="custom.css"),
      tags$style(HTML('
                                /* logo */
                                .skin-blue .main-header .logo {
                                background-color: #000033;
                                }
                                
                                /* navbar (rest of the header) */
                                .skin-blue .main-header .navbar {
                                background-color: #000033;

                                '))
    ),
    tabItems(
      
      # Intro
      tabItem(
        tabName = "Intro",
              div(style='overflow-y:scroll'),
              mainPanel(
                box(width=12, h1("Airlines Flight Statistics Widget"), background='black',
               
                    br(),
                    h3("Introduction:"),
                    h4("The inspiration for this dashboard came from The Economist's Big Mac Index and
                       the idea of story telling using a dashboard. With limited shinydashboard
                       knowledge, this represents my first attempt at building a dashboard. This app was
                       designed to visualize the flight delays in the U.S. in 2008 to better understand airline
                       performance and to understand the seasonality or trends within the airline industry."),
                    br(),
                    h3("Overview and Guide:"),
                    h4("This dashboard contains 3 tabs: 'Summary', 'Delays', and 'Map'."),
                    tags$ul(
                      tags$li(tags$b("Summary: Displays the hourly, daily, and monthly trend of flights. It also displays
                                        the total number of flights and total distance travelled (miles) for a selected airline.")), 
                      tags$li(tags$b("Delays: Displays a bar chart showing which type of delays contributes most to the overall delay,
                              a lollipop plot that displays the top 10 (origin) airport where that airlines experiences the most delay,
                              and a heatmap that shows the total number of delays of any day within the year. Lastly, I also showed
                              the number of delays (in minutes) and the average delay per every delayed flight for the selected airline")), 
                      tags$li(tags$b("Map: A geographical map that shows the top 10 most delayed routes for the selected airport! A table is also provided
                              to display the delay percentages for each airport and it's destination airport."))
                    )
                )
              )
      ),
      
      #Summary tab
      tabItem(tabName = "Summary",
              fluidRow(
                              box(width=4,status="primary",
                                  h1(textOutput("selected_var")),br(),
                                  h4("Total Number of Flights in 2008: "),h4(textOutput("totalflights")),br(),
                                  h4("Total Distance Travelled (miles) in 2008: "),h4(textOutput("totaldist")),br(),
                                  h4("Tip: Use the dropdown menu on the left side bar to change airlines!")
                                  ),
                              
                              box(width=6,
                                  plotOutput("barcharthour", height = 282, width = 550),
                                  plotOutput("barchartday", height = 282, width = 550),
                                  plotOutput("barchartmonth", height = 282, width = 550)
              ))
      ),
      #Delays tab
      tabItem(tabName = "Delays",
              fluidRow(
                box(width=6,align='Left',
                    h1(textOutput("selected_var1")),br(),
                    h4("Total Delays (mins): "),h4(textOutput("totaldelaymins")),br(),
                    h4("Average delay (mins) for every delayed flight: "),h4(textOutput("avgdelaymins")),br(),
                    plotOutput("typedelays", height = 486, width = 486)
                ),
                
                box(width=6, align='Right',
                    plotOutput("arrdelay", height = 282, width =486),
                    plotOutput("delayheat", height = 486, width = 486)
                    
                
            ))
      ),
      
      #Map tab
      tabItem(tabName = "Map",
              fluidRow(
                column(width = 9,
                       box(width = NULL, solidHeader = TRUE, background = "navy",
                           leafletOutput("omap", height = 400), # US map
                           tags$style(HTML("
                           
                           .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate, .dataTables_wrapper .dataTables_paginate .paginate_button.current:hover{
                           color: #ffffff;
                           }
                           
                           .dataTables_wrapper .dataTables_paginate .paginate_button{box-sizing:border-box;display:inline-block;min-width:1.5em;padding:0.5em 1em;margin-left:2px;text-align:center;text-decoration:none !important;cursor:pointer;*cursor:hand;color:#ffffff !important;border:1px solid transparent;border-radius:2px
                           }
                           
                           .dataTables_length select {
                           color: #0E334A;
                           }
                           
                           .dataTables_filter input {
                            color: #0E334A;
                           }
                           
                           thead {
                           color: #ffffff;
                           }
                           
                           tbody { color: #000000;}"
                           )),
                    
                           dataTableOutput("table") # Corresponding data table
                       )
                ),
                column(width = 3,
                       box(width = NULL, status = "warning",background = "black",
                           selectInput("airport", "Airport Letter Code", selected = 'LAX',
                                       airport_codes),
                           selectInput("sort_choice", "Sorting Factor",
                                       choices = c('Total Delays', 'Percentage')),
                           sliderInput('months', 'Month Range',min=1, max=12,
                                       value=c(1,12))),
                       h4("Pro Tip: Type in the IATA code for any airport in \"Airport Letter Code\" that you'd like to see and avoid having to
                          scroll the list of airports!"),
                    
                       h4("You can also click on the red lines for more information regarding that route!")
                      )
                  )
            )
      )
    )
  )
