library(feather)
library(tidyverse)


path_simulations <- "../../data/simulations"

simulations_ball <- read_feather(str_c(path_simulations, "/sim_ball.feather"))
simulations_ball %>% glimpse()
simulations_ball %>% names()

one_simulation <- simulations_ball %>% 
  filter(simulation == "sim_0") %>% 
  mutate(run = as.factor(run))

ggplot(one_simulation, aes(px, py, color = run))+
  # geom_point(alpha = .2, size = .5)+
  geom_path() +
  scale_x_continuous(limits = c(0, 600), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 700), expand = c(0, 0)) +
  theme_void()+
  theme(legend.position='none')
  
