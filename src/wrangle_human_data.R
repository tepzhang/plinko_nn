library(tidyverse)


## human experiment data
path_tobi_data <- "C:/Users/tepzh/Dropbox/Stanford/CS229 machine learning/project/plinko_traking_Tobi/plinko_tracking/code/R/plinko_eye_tracking_data.RData"
path_data_nn <- "C:/Users/tepzh/Dropbox/Stanford/CS229 machine learning/project/plinko_nn/data"

df.data <- load(path_tobi_data)

# write the data to csv
write_csv(df.data %>% filter(experiment == "vision"), 
          str_c(path_data_nn, "/experiment1/eye_tracking_data.csv"))
write_csv(df.data %>% filter(experiment == "vision_sound"), 
          str_c(path_data_nn, "/experiment2/eye_tracking_data.csv"))
