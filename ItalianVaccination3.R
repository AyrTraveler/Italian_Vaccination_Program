#### 0 - LOADING LIBRARIES AND CREATING THE DATASET ####

#This project aims to build an ARIMA model to forecast the daily covid-19 vaccination rate in Italy.
#ARIMA stands for Auto Regressive Integrated Moving Average, and is a common approach used in time series forecasting.
#ARIMA will be compared with a polynomial regression, using MAPE (mean average percentual error) as evaluation function.
#We will use cross validation to observe the performance of the model. How?
#The MAPE gives a measure of how much the prediction is far from the actual daily number of doses.
#We will then calculate the MAPE using the 15-day forecast and the true number of doses of the latest 14 days of the original dataset
#The original data is provided by Our World in Data, and is available in csv format at: https://www.kaggle.com/arthurio/italian-vaccination
#This csv file contains day-by-day number of first and second doses, with the level of granularity expressed by: 

# 1. Age-group (16-19, 20-30, 30-40,40-50,50-60,60-70,70-80,80-90,90+)
# 2. Italian Region (there are 21 different regions)
# 3. Date of vaccination (starting from December 27th 2020)
# 4. Vaccine Supplier (Pfizer, Astrazeneca, Moderna and Janssen)

#Section 0 will focus on dataset creation, data-cleaning and normalization, train-set and test-set creation
#Section 1 will try to explore the dataset in order to see how the above features 1. 2. 3 and 4. affect the vaccines distribution.
#Section 2 will build the (optimal) ARIMA model.
#Section 3 will evaluate the predicted vaccine rate with the test set, and compare the resulting MAPE with a polynomial interpolation regression

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(tseries)) install.packages("tseries", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")
if(!require(gtools)) install.packages("gtools", repos = "http://cran.us.r-project.org")
if(!require(sqldf)) install.packages("sqldf", repos = "http://cran.us.r-project.org")

library(dplyr)
library(gtools)
library(tidyverse)
library(ggplot2)
library(caret)
library(tseries)
library(forecast)
library(MLmetrics)
library(sqldf)

#the dataset used for this project is available at the public repository:
url <- "https://github.com/AyrTraveler/Italian_Vaccination_Program/raw/main/italian_vaccination.csv"
#If you want an updated version of the dataset, you can find it here: https://www.kaggle.com/arthurio/italian-vaccination

dat <- read_csv(url)
names(dat)

#we throw away 2020 observations for the sake of simplicity. Why?
#Only a few dozen of people have been vaccinated from 2020-12-27 to 2020-12-31
dat  <- dat %>% filter(administration_date > "2020-12-31")

# how the original dataset looks like:
temp <- dat %>% 
  select(administration_date,supplier,age_range,females,males,first_dose,second_dose,region_name)
knitr::kable(head(temp))



# We only take the information we will focus on in Section 1: Data Exploration
# The daily doses is obtained as the sum of: FIRST + SECOND doses:
# this is going to be the actual time series we are going to predict.

dat <- dat %>% 
  group_by(administration_date,region,supplier, region_name, age_range) %>% 
  summarise(dose = first_dose+second_dose, males = sum(males), females = sum(females))


#TRAIN SET: all the observations from Jan 1st up to max(date) - 14
#TEST SET: all the observations from max(date) - 14 up to max(date) 

test_set <- dat %>%  filter(administration_date >= max(dat$administration_date) -14)
train_set <- dat %>%  filter(administration_date < max(dat$administration_date) -14)


#we categorize the regions in three zones: Northern Italy - Central Italy - Southern Italy 
regions  <- train_set %>% pull(region) %>%
  as.data.frame() %>%
  distinct() %>% 
  setNames(c("region")) %>% 
  mutate(area = case_when(
    
    region %in% c("ABR", "LAZ","UMB","EMR","TOS","MAR") ~ "Central",
    region %in% c("SAR", "SIC","PUG","BAS","CAL","CAM","MOL") ~ "Southern",
    region %in% c("PIE", "PAT","PAB","FVG","VEN","LOM","LIG","VDA") ~ "Northern"
  )
  )

#extend the dataset to include the "Area" information
train_set <- train_set  %>% left_join(regions, by = "region")
test_set <- test_set  %>% left_join(regions, by = "region")

#### 1 - DATA EXPLORATION ####

##  Vaccination rate by age-range

# In Italy, the first people to receive a vaccine shot in early 2021 were health-care workers, immediately followed by older people (> 80 years old).\
# The plot below shows how the vaccine daily rate evolved over the months:
  

#daily doses distribution per area
dat %>% group_by(administration_date,age_range) %>%
  summarize(dose = sum(dose)) %>% 
  ggplot(aes(administration_date,dose/100000, col = age_range)) +
  geom_line(size = 0.2) +
  geom_point(size = 0.5) +
  geom_smooth() +
  labs(y= "Daily doses / 1e+5", x = "") 



# We can see that now (late May 2021) most of the people older than 70 years old have already received the vaccine.\
# We also note an increasing trend for all the remaining age groups.\
# If we sum the vaccine doses day by day we can plot the cumulative vaccine trend by age group:\ 


cumulativeSum = function(x){
  
  cumsum(ifelse(is.na(x),0,x))
  
}


clean_set  =  dat %>%  group_by(administration_date, age_range) %>%
  summarize(dose = sum(dose))
clean_set <- clean_set %>% spread(age_range,dose)
ADMIN_DATE <- as.data.frame(clean_set) %>% select(administration_date)
clean_set <- as.data.frame(clean_set) %>% select(-administration_date)
clean_set <- apply(as.matrix(clean_set),2,cumulativeSum)
clean_set <- as.data.frame(clean_set)  %>% gather(age_range, dose, `16-19`:`90+`)
clean_set <- data.frame(ADMIN_DATE,clean_set)

clean_set %>% ggplot(aes(administration_date,dose/1000000, col = age_range)) +
  geom_line(size = 0.2) +
  geom_point(size = 0.5) +
  geom_smooth() +
  scale_y_continuous(position = "right") +
  theme(legend.position = "left") +
  labs(y= "total doses [millions]", x = "") 


# Finally, the barplot below summarizes the total up to date vaccine shots by age group:\

clean_set %>% 
  group_by(age_range) %>% 
  summarise(dose = max(dose)/1000000) %>%
  ggplot(aes(age_range,dose))+
  geom_bar(stat="identity", show.legend = FALSE, fill = "#00BFC4") +
  labs(y= "Doses [millions]", x = "Age Range") 


## Vaccination rate by  region and area

# Italy is divided into 21 regions (Abruzzo, Basilicata, Calabira, Campania, Emilia ROmagna, Friuli-venezia-giulia, Lazio, Liguria, Lombardia, Marche, Molise, Provincia di Bolzano, Provincia di Trento, Piemonte, Puglia, Sardegna, Sicilia, Toscana, Umbria, Valle d'Aosta, Veneto).\

# The plot below shows:\
# 
# - the current number of vaccine shots by region, in red
# - the total population of the region, in blue
# 
# Considering that most of vaccines requires two shots to receive a complete vaccination,
# at the end of the vaccination program an ideal scenario would be to have the red bar to be twice the blue bar in length.

train_set %>% group_by(region,area) %>% 
  summarize(dose = sum(dose)) %>% 
  mutate( population =
            case_when(
              
              region == "ABR" ~ 1.293941,
              region == "BAS" ~ 0.553254,
              region == "CAL" ~ 1.894110,
              region == "CAM" ~ 5.712143,
              region == "EMR" ~ 4.464119,
              region == "FVG" ~ 1.206216,
              region == "LAZ" ~ 5.755700,
              region == "LIG" ~ 1.524826,
              region == "LOM" ~ 10.027602,
              region == "MAR" ~ 1.512672,
              region == "MOL" ~ 0.300516,
              region == "PAB" ~ 0.533349,
              region == "PAT" ~ 0.545497,
              region == "PIE" ~ 4.311217,
              region == "PUG" ~ 3.953305,
              region == "SAR" ~ 1.611621,
              region == "SIC" ~ 4.875290,
              region == "TOS" ~ 3.692555,
              region == "UMB" ~ 0.870165,
              region == "VDA" ~ 0.125034,
              region == "VEN" ~ 4.879133,
              
            ),
          dose = dose/1000000
  ) %>%
 gather(kind,dose,`dose`:`population`) %>% 
  mutate(dose = round(dose,2)) %>% 
   ggplot(aes(reorder(region,dose) ,dose, fill = kind )) +
   geom_bar(stat="identity", position = position_dodge(), show.legend = TRUE) +
  coord_flip() +
   labs(y= "Doses [millions]", x = "Regions") 

# Different regions have different populations. In order to see if the vaccination trend is homogeneous across all the country, we can categorize the Italian regions to fall in three different areas: 
# 
# - Northern Italy
# - Central Italy
# - Southern Italy
# 
# The plot below shows that the trend of daily vaccine shots is homogeneous across the three different zones 
# (Northern Italy is slightly higher, but we need to keep in mind that most of Italian people live in this zone)


#daily doses distribution per area
train_set %>% group_by(administration_date,area) %>%
  summarize(dose = sum(dose)/100000) %>% 
  ggplot(aes(administration_date,dose, col = area)) +
   geom_line(size = 0.2) +
   geom_point(size = 0.5) +
  geom_smooth() +
  labs(y= "Daily doses / 1e+5", x = "") 


## Vaccination rate by vaccine supplier

# Most of the doses come from Pfizer/Biontech, followed by Astrazeneca and Moderna. Janssen is starting only now (May 2021) to show a positive trend. We can see that from the plot:

#daily doses distribution per supplier
train_set %>% group_by(administration_date,supplier) %>%
  summarize(dose = sum(dose)/100000) %>% 
  ggplot(aes(administration_date,dose, col = supplier)) +
  geom_line(size = 0.2) +
  geom_point(size = 0.5) +
  geom_smooth() +
  labs(y= "Daily doses / 1e+5", x = "") 


## Vaccination rate by sex

# The following plot shows that the female daily vaccine rate trend is slightly higher compared to the male trend.\
# This is probably due to the fact that the first people who received a shot fall on older category groups, where females are generally more then males (and this is probably true also in other European countries).

  dat %>% 
  group_by(administration_date) %>% 
  summarize(males=sum(males),females=sum(females)) %>% 
  gather(sex, dose, `males`:`females`) %>% 
  ggplot(aes(administration_date,dose/100000, col = sex)) +
  geom_line(size = 0.2) +
  geom_point(size = 0.5) +
  geom_smooth() +
  labs(y= "Daily doses / 1e+5", x = "") 


#### 2 - MODELING ####

# 2.1 TRANSFORMING THE DATASET INTO A TIME SERIES
  
#A time series is a list of observations at different times. The time frequency can be yearly, monthly, daily or even less than a day.
#For this project we focus on the daily vaccination rate. So, the code below transform the train_set into a time series.
#To do so, we simply group by the administration_date and summarize the doses as the sum of doses.
#In this way we lose the information about the other features (vaccine supplier, age range, sex, regions), 
#and we try to build an ARIMA model to predict the future daily vaccination rate 
#using only the data from the past observations of the time series.
  
daily_doses <- train_set %>%
  group_by(administration_date) %>% 
  summarise(dose = sum(dose)) %>%
  pull(dose)


#An ARIMA model depends on 3 parameters:
  
  # d, the order of differencing
  # p, the number of lagged observation used to build the AR part of the model
  # q, the number of lagged errors used to build the MA part of the model

#to build an ARIMA model we only need to pick the values for d,p,q
#and call the function ARIMA(time_series, order = c(d,p,q)) from the tseries package
  
#sections 2.2, 2.3, 2.4 describe how to choose the set of d,p,q parameters to get the optimal ARIMA regression.

    
# 2.2 The d parameter: Order of differencing
adf.test(daily_doses, alternative = "stationary")
acf(daily_doses)

first_diff = diff(daily_doses)
adf.test(first_diff, alternative = "stationary")
acf(first_diff)

## 2.3 Choosing the p & q PARAMETERS for ARIMA model

#p parameter: p is the number of lagged observations for the AR part of the model
#Partial auto-correlation function useful to select the right p number
pacf(first_diff)
PACF = pacf(first_diff)
PACF = data.frame(correlation = PACF$acf) %>% mutate(n= row_number()) %>% filter(abs(correlation) < 0.1)
PACF

#q parameter: q is the number of lagged errors for the MA part of the model
#auto-correlation function useful to select the right q number
acf(first_diff)
ACF = acf(first_diff)
ACF = data.frame(correlation = ACF$acf) %>% mutate(n= row_number()) %>% filter(abs(correlation) < 0.1)
ACF

## 2.4 fit the best model with lowest AIC

#the cross join is used to create all the combinations of feasible P-Q parameters
PQ <- sqldf('SELECT PACF.n, ACF.n FROM  PACF CROSS JOIN ACF') %>% setNames(c("P","Q"))

#the below while-construct loops all the p-q combinations of the PQ dataset and computes the AIC value
#AIC: Aiki Information Criterion
#AIC is a tool for performing a multi-objetcive optimization: we need to achieve 2 optimal results at the same time:
#1. to choose the p-q parameters providing the best fit regression
#2. to minimize over-training 
#the lowest value of AIC provides the optimal p-q combination that respects both 1. and 2.
#For some p-q combinations, the ARIMA(p,d,q) function may raise an error due to non-stationarity:
#the trycatch construct allows the while-loop to continue even when an error is raised

temp = data.frame()
i = 1
PREV = -1
while (i < length(PQ$P)){
  
    tryCatch({
      model <- arima(daily_doses, order = c(PQ$P[i],1,PQ$Q[i]))  
      
    }, warning = function(w) {
    }, error = function(e) {
    }, finally = {})
  
  temp <- bind_rows(temp,data.frame(ifelse(PREV == model$aic,1000000,model$aic) ,PQ$P[i] , PQ$Q[i]))
  
  PREV = model$aic
  
  i = i + 1
  print(i)
}

temp <- temp %>% setNames(c("AIC","P","Q")) %>%  filter(AIC < 10000) %>% 
        mutate(n = row_number())


#Now we can pick the optimal p-q (p = 9, q = 9)
min(temp$AIC) #lowest AIC: 2954
p <-temp$P[which.min(temp$AIC)]
q <-temp$Q[which.min(temp$AIC)]

p = 9
q = 9

#optimal ARIMA: p = 9, d = 1, q = 9
model <- arima(daily_doses, order = c(p,1,q))


#we now try to forecast the vaccine daily rate for a 14-day time span
#Then we plot the original daily trend - day 0 to 129 (blue)
#versus the predicted daily trend - day 130 to 144 (red)
# day 0 is Jan 1st 2021
y_hat <- predict(model,n.ahead = 15)
original = data.frame(daily_doses, TAG = "original")
fcast = data.frame(daily_doses = y_hat$pred, TAG = "forecast")

#y axis is expressed in hundred of thousands doses
#x axis is the number of days starting from Jan 1st 2021
final = bind_rows(original,fcast) %>% mutate(nn = row_number())
xcept = max(final$nn)-14
final %>%
  mutate(daily_doses = daily_doses/100000) %>% 
  ggplot(aes(nn,daily_doses, col = TAG)) +
  geom_point(size = 0.5)+
  geom_line(size = 0.2)+
  geom_vline(xintercept = xcept, linetype="dotted", color = "red", size=1.5) +
  labs(y= "Doses / 1e5", x = "") 


#### 3 - EVALUATION ####

# In this section, we try to evaluate the model by measuring the MAPE (mean average percentage error) 
# between the predicted daily doses from the ARIMA and the true doses from the test set
# The resulting MAPE will be compared with the MAPE obtained using a more simple regression:
# we try to fit the daily vaccine doses with a cubic polynomial ax^3 + bx^2 + cx + d


results = data.frame()


# 3.1 ARIMA FORECAST EVALUATION

# pulling the time series from the test set:
true_dose <- test_set %>% 
  group_by(administration_date) %>% 
  summarise(dose = sum(dose)) %>%
  pull(dose)

# filtering the 14 days forecast from the final dataset:
predicted_dose <- final %>% filter(TAG == "forecast") %>% pull(daily_doses)

# calculating the MAPE (using the function from MLmetrics library):
MAPE(true_dose,predicted_dose)

# keep the result, because later we want to compare it with the MAPE of the poly regression:
results = bind_rows(results, data.frame(Method = "ARIMA(9,1,9)", MAPE = MAPE(true_dose,predicted_dose)))

#the plot shows how well the forecast fits the original series:
t <- data.frame(dose = true_dose, TAG = "true") %>% mutate(nn = row_number())
f <- data.frame(dose = predicted_dose, TAG = "forecast") %>% mutate(nn = row_number())
outcome =  bind_rows(t,f) 
outcome %>% ggplot(aes(nn,dose,col = TAG)) + geom_line(size =0.8)


# 3.2 CUBIC FIT FORECAST EVALUATION

total_daily_doses <- dat %>% 
  group_by(administration_date) %>% 
  summarise(dose = sum(dose)) %>% 
  mutate(nn = row_number())

DSET = data.frame(dose = total_daily_doses$dose) %>%  mutate(n = row_number())

# building the polynomial residuals using the lm() function:
modelPoly <- lm(DSET$dose ~ poly(DSET$n,3))

# building the actual polynomial regression ax^3 + bx^2 + cx + d:
forecast_seq <- seq(130,144)

cubic_fit = function(n) {

  pred =   modelPoly$coefficients[1] +
    modelPoly$coefficients[2]*n +
    modelPoly$coefficients[3]*n^2 +
    modelPoly$coefficients[4]*n^3
  pred
}


# plotting the original series together with the ARIMA forecast and the polynomial forecast

y = data.frame(dose = -sapply(forecast_seq, cubic_fit)/1000000, TAG = "cubic") %>% mutate(nn = row_number())
outcome =  bind_rows(outcome,y) 
outcome %>% ggplot(aes(nn,dose,col = TAG)) + geom_line(size =0.8)
predicted_dose <- outcome  %>% filter(TAG == "cubic") %>% pull(dose)

# writing the resulting MAPE
results = bind_rows(results, data.frame(Method = "Cubic Poly regression", MAPE = MAPE(true_dose,predicted_dose)))

#comparison between ARIMA and Poly regression:
knitr::kable(results)

#final MAPE is about 7%




