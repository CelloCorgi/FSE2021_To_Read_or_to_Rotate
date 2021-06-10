source("https://gist.githubusercontent.com/benmarwick/2a1bb0133ff568cbe28d/raw/fb53bd97121f7f9ce947837ef1a4c65a73bffb3f/geom_flat_violin.R")

#import appropriate libraries
library(tidyverse, ggthemes)
library(lme4)
library(lmerTest)
library(lmtest)
library(sensemakr)
library(ggfortify)
library(dplyr)
library(MASS)
library(car)

#set working directory
setwd("/Users/mfansher/Box/Spatial_Reasoning/")

#Clean the data, calculate accuracy scores for reading, programming, and spatial ability measures
data = read_csv("SPC.csv") %>%
  mutate(id = 1:n()) %>%
  filter(AttendedPosttest == TRUE & NumTrainingSessions >= 6 &
           !is.na(FeelHelped) & !is.na(EffectedPlans)) %>%
  mutate(PreYoon = PreYoon/30, PrePFT = Pre/20, PostPFT = Post/20,
         PreRead = PreReadII/25, PostYoon = PostYoon/30, PostRead = PostRead/25,
         PreProg = PreProg/12,
         PostProgOld = PostProgI/12, 
         PostProgNew = (PostProgII-PostProgI)/15, 
         PostProgBoth = PostProgII/27,
         EngNat = ifelse(grepl("engl",Language), 1,
                         ifelse(grepl("Engl", Language), 1, 0)), 
         FeelHelped = ifelse(FeelHelped == "No", "No", 
                             ifelse(FeelHelped == "Yes", "Yes", "Unsure")))
#_______________________________________________________________________________________
#This document provides the model specification for each of the multiple linear regression models reported. 
#For each of the models reported below, if you wish to test that the assumptions of the model are not violated, 
#copy and paste the code below, replacing "model" with the name of the model that you wish to test. 

##Check statistical assumptions of multiple regression

#testing for independence of observations
dwtest(#model formula here,
  data = data)

#check for homoscedasticity
ncvTest(model)

#shapiro wilk normality test
sresid <- studres(model)
shapiro.test(sresid)

#check for multicolinearity by inspecting VIFs
car::vif(model)

#check for linearity, outliers, and visually inspect normality
#residuals vs fitted plot, residuals vs leverage plot w Cook's distance, normal q-q plot
plot(model)
#_______________________________________________________________________________________

#RESULTS

#training effectiveness validation

#dv = yoon post test, iv = yoon pre-test, trainingtype 
#run model
model1 <- lm(data = data, PostYoon ~ PreYoon+TrainingType)
summary(model1)


#dv = reading post test, iv = reading pre-test, trainingtype 
#run model
model2 <- lm(data = data, PostRead ~ PreReadII+TrainingType)
summary(model2)


#dv = pft post test, iv = pft pre-test, trainingtype 
#run model
model3 <- lm(data = data, PostPFT ~ PrePFT+TrainingType)
summary(model3)

#_______________________________________________________________________________________

#RQ1.  Does reading training improve programming abilities?

#dv = programming post-test, iv = programming pre-test, training type
#run model
model4 <- lm(data = data, PostProgBoth ~ PreProg+TrainingType)
summary(model4)
#calculate the size of the effects (cohens f2)
partial_f2(model4)

#_______________________________________________________________________________________

#RQ2.
#We analyze if the Reading Treatment’s benefit varies based on programming question type

#dv = code completion acc @ post-test
#iv = code completion acc @ pre-test, training type
#run model
model5 <- lm(data = data, PostCode ~ Code_Comp+TrainingType)
summary(model5)



#dv = definitional acc @ post-test
#iv = definitional acc @ pre-test, training type
#run model
model6 <- lm(data = data, PostDef ~ PreDefinitional+TrainingType)
summary(model6)


#dv = tracing acc @ post-test
#iv = tracing acc @ pre-test, training type
#run model
model7 <- lm(data = data, PostTrace ~ PreTracing+TrainingType)
summary(model7)
#calculate the size of the effects (cohens f2)
partial_f2(model7)

#_______________________________________________________________________________________
#RQ3.
#participant subpopulation effects

#dv = programming acc @ post-test
#iv = preyoon, training type, preyoon*training type, programming pretest score
#run model
model10 <- lm(data = data, PostProgBoth ~ PreYoon*TrainingType+PreProg)
summary(model10)


#dv = programming acc @ post-test
#iv = prePFT, training type, prePFT*training type, programming pretest score
#run model
model11 <- lm(data = data, PostProgBoth ~ PrePFT*TrainingType+PreProg)
summary(model11)


#dv = programming acc @ post-test
#iv = preReading, training type, preReadingtraining type, programming pretest score
#run model
model12 <- lm(data = data, PostProgBoth ~ PreRead*TrainingType+PreProg)
summary(model12)


#dv = programming acc @ post-test
#iv = native english speaker, training type, english speaker*training type, programming pretest score
#run model
model13 <- lm(data = data, PostProgBoth ~ EngNat*TrainingType+PreProg)
summary(model13)


#dv = programming acc @ post-test
#iv = gender, training type, gender*training type, programming pretest score
#run model
model14 <- lm(data = data, PostProgBoth ~ Gender*TrainingType+PreProg)
summary(model14)

#_______________________________________________________________________________________

#Incoming Spatial Ability compared to other studies
t.test(data$PrePFT, mu = .55, alternative = "two.sided")
t.test(data$PreYoon, mu = .505, alternative = "two.sided")

#RQ5 Free analysis response
tbl = table(data$EffectedPlans, data$TrainingType) 
chisq.test(tbl) 

tbl2 = table(data$FeelHelped, data$TrainingType) 
chisq.test(tbl2) 

#Reproduce Figures

data %>%
  pivot_longer(cols=c(PreProg,PostProgBoth),
               names_to = "time",
               values_to = "score") %>%
  mutate(time = ifelse(time=="PreProg","Pre-test","Post-test"),
         time = factor(time,unique(time))) %>%
  ggplot(aes(x=time, y=score, colour=TrainingType, group=TrainingType)) +
  stat_summary(position=position_dodge(.1),geom="line") +
  stat_summary(position=position_dodge(.1)) +
  geom_point(position=position_jitterdodge(jitter.width = .5,dodge.width = .1),alpha = .75, size = .4) +
  scale_colour_brewer(palette = "Set1") +
  scale_x_discrete("Time") +
  scale_y_continuous("Score")+
  theme_classic(base_size = 12)+
  theme(legend.position = "none")
ggsave("prog.pdf", units="in", height = 3, width = 3)

data %>%
  pivot_longer(cols=c(PreTracing,PostTrace),
               names_to = "time",
               values_to = "score") %>%
  mutate(time = ifelse(time=="PreTracing","Pre-test","Post-test"),
         time = factor(time,unique(time))) %>%
  ggplot(aes(x=time, y=score, colour=TrainingType, group=TrainingType)) +
  stat_summary(position=position_dodge(.1),geom="line") +
  stat_summary(position=position_dodge(.1)) +
  geom_point(position=position_jitterdodge(jitter.width = .5,dodge.width = .1),alpha = .75, size = .4) +
  scale_colour_brewer(palette = "Set1") +
  scale_x_discrete("Time") +
  scale_y_continuous("Score")+
  theme_classic(base_size = 12)+
  theme(legend.position = "none")
ggsave("trace.pdf", units="in", height = 3, width = 3)

data %>%
  pivot_longer(cols=c(PreDefinitional,PostDef),
               names_to = "time",
               values_to = "score") %>%
  mutate(time = ifelse(time=="PreDefinitional","Pre-test","Post-test"),
         time = factor(time,unique(time))) %>%
  ggplot(aes(x=time, y=score, colour=TrainingType, group=TrainingType)) +
  stat_summary(position=position_dodge(.1),geom="line") +
  stat_summary(position=position_dodge(.1)) +
  geom_point(position=position_jitterdodge(jitter.width = .5,dodge.width = .1),alpha = .75, size = .4) +
  scale_colour_brewer(palette = "Set1") +
  scale_x_discrete("Time") +
  scale_y_continuous("Score")+
  theme_classic(base_size = 12)+
  theme(legend.position = "none")
ggsave("def.pdf", units="in", height = 3, width = 3)

data %>%
  pivot_longer(cols=c(Code_Comp,PostCode),
               names_to = "time",
               values_to = "score") %>%
  mutate(time = ifelse(time=="Code_Comp","Pre-test","Post-test"),
         time = factor(time,unique(time))) %>%
  ggplot(aes(x=time, y=score, colour=TrainingType, group=TrainingType)) +
  stat_summary(position=position_dodge(.1),geom="line") +
  stat_summary(position=position_dodge(.1)) +
  geom_point(position=position_jitterdodge(jitter.width = .5,dodge.width = .1),alpha = .75, size = .4) +
  scale_colour_brewer(palette = "Set1") +
  scale_x_discrete("Time") +
  scale_y_continuous("Score")+
  theme_classic(base_size = 12)+
  theme(legend.position = "none")
ggsave("code.pdf", units="in", height = 3, width = 3)

data %>%
  mutate(N = n()) %>%
  mutate(FeelHelped = factor(FeelHelped,unique(FeelHelped))) %>%
  group_by(TrainingType, FeelHelped) %>%
  summarise(percent = (n()/N) * 100) %>%
  ggplot(aes(x=FeelHelped, y=percent, fill=TrainingType, group=TrainingType, colour = TrainingType))+
  geom_col(width=.7,position = position_dodge(.72)) +
  scale_fill_brewer(palette = "Set1")+
  scale_colour_brewer(palette = "Set1")+
  scale_x_discrete("Response")+
  scale_y_continuous("Percent of Participants")+
  theme_get()+
  theme(legend.position = "none",
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))
ggsave("helped.pdf", units="in", height = 3, width = 3)

data %>%
  mutate(N = n()) %>%
  mutate(EffectedPlans = factor(EffectedPlans,unique(EffectedPlans))) %>%
  group_by(TrainingType, EffectedPlans) %>%
  summarise(percent = (n()/N) * 100) %>%
  ggplot(aes(x=EffectedPlans, y=percent, fill=TrainingType, group=TrainingType, colour = TrainingType))+
  geom_col(width=.7,position = position_dodge(.72)) +
  scale_fill_brewer(palette = "Set1")+
  scale_colour_brewer(palette = "Set1")+
  scale_x_discrete("Response")+
  scale_y_continuous("Percent of Participants")+
  theme_get()+
  theme(legend.position = "none",
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=8))
ggsave("plans.pdf", units="in", height = 3, width = 3)

